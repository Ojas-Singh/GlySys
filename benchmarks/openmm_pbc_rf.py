#!/usr/bin/env python3
"""Layered PBC explicit-water parity against OpenMM's Reference platform.

Input: JSON from `cargo run -p glysys-dynamics --example explicit_reference`.

Layer 1 (statics): total energy, per-term components (OpenMM force groups;
an LJ-only leg with zeroed charges isolates electrostatics/RF), and forces
on the minimized (A) and post-NVE (B) snapshots, rigid and flexible.
Layer 2 (NVE): total-energy drift magnitudes ours-vs-OpenMM at matched
timestep, flexible leg isolating constraints, and dt-convergence scaling.
Layer 3 (NVT): kinetic-energy means (DOF-convention free), potential means,
and production temperature with block-averaged standard errors.

Tolerances live in TOL below; rationale in benchmarks/README.md (parity
section). Exits nonzero on any failure; always prints the full report JSON.
"""
import json, sys, tempfile, math
from pathlib import Path

TOL = {
    "energy_abs_kcal_mol": 0.05,
    "term_abs_kcal_mol": 0.05,
    "force_abs_kcal_mol_A": 0.02,
    "nve_drift_per_atom": 0.05,
    "nvt_temperature_K": 8.0,
    "nvt_energy_kcal_mol": 10.0,
}

r = json.load(open(sys.argv[1]))
elec = r["electrostatics"]
assert elec["method"] == "reaction-field"
cutoff_nm = elec["cutoffAngstrom"] / 10.0
rf_dielectric = elec["solventDielectric"]
KB = 0.00198720425864083

import openmm as mm
from openmm import app, unit
import numpy as np

failures = []
report = {"schemaVersion": 1, "openmmVersion": mm.__version__, "tolerances": TOL}

with tempfile.TemporaryDirectory() as folder:
    Path(folder, "system.prmtop").write_text(r["files"]["system.prmtop"])
    prmtop = app.AmberPrmtopFile(str(Path(folder, "system.prmtop")))

    topo_atoms = list(prmtop.topology.atoms())

    def make_system(rigid_water):
        system = prmtop.createSystem(
            nonbondedMethod=app.CutoffPeriodic,
            nonbondedCutoff=cutoff_nm * unit.nanometer,
            # GlySys's Settle model constrains both solute X-H bonds and all
            # three distances in each TIP3P water. Build the same topology
            # explicitly so OpenMM can identify the water triangle as a
            # SETTLE cluster while HBonds supplies the solute constraints.
            constraints=app.HBonds if rigid_water else None,
            rigidWater=False,
            removeCMMotion=False,
        )
        if rigid_water:
            # HBonds already adds the two O-H distances. Add H-H to close the
            # triangle; ReferenceConstraints then dispatches the water through
            # its analytic SETTLE implementation. The solute X-H constraints
            # remain in the CCMA side, matching GlySys's general RATTLE path.
            for res in prmtop.topology.residues():
                if res.name not in ("HOH", "WAT"):
                    continue
                hs = [a.index for a in res.atoms()
                      if a.element.symbol == "H"]
                assert len(hs) == 2, f"non-TIP3P water residue {res}"
                # Ideal TIP3P H-H (2*0.9572*sin(104.52/2) A); prmtop files
                # carry no coordinates. Matches minimized-geometry targets
                # to ~1e-3 A (manifold shift negligible at our tolerances).
                d = 2 * 0.09572 * math.sin(math.radians(104.52) / 2)
                system.addConstraint(hs[0], hs[1], d)
        nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
        nb.setUseSwitchingFunction(False)
        nb.setReactionFieldDielectric(rf_dielectric)
        nb.setUseDispersionCorrection(False)
        groups = {}
        for f in system.getForces():
            name = f.__class__.__name__
            g = {"HarmonicBondForce": 0, "HarmonicAngleForce": 1,
                 "PeriodicTorsionForce": 2, "NonbondedForce": 3}.get(name, 4)
            f.setForceGroup(g)
            groups.setdefault(g, name)
        return system, nb

    def zero_charges(system, nb):
        for i in range(nb.getNumParticles()):
            q, sig, eps = nb.getParticleParameters(i)
            nb.setParticleParameters(i, 0.0, sig, eps)
        for i in range(nb.getNumExceptions()):
            a, b, qq, sig, eps = nb.getExceptionParameters(i)
            nb.setExceptionParameters(i, a, b, 0.0, sig, eps)

    to_nm = lambda ps: [mm.Vec3(p["x"] / 10.0, p["y"] / 10.0, p["z"] / 10.0) for p in ps]
    if "boxAngstrom" in r:
        lx, ly, lz = (v / 10.0 for v in r["boxAngstrom"])
        box = (mm.Vec3(lx, 0, 0), mm.Vec3(0, ly, 0), mm.Vec3(0, 0, lz))
    else:
        box = prmtop.topology.getPeriodicBoxVectors()
    n_atoms = len(r["snapshotA"])
    system, nb = make_system(rigid_water=True)

    def run_points(system, coords, groups=(0, 1, 2, 3)):
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
        context.setPositions(to_nm(coords))
        if box is not None:
            context.setPeriodicBoxVectors(*box)
        out = {}
        for g in groups:
            e = context.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(
                unit.kilocalories_per_mole)
            out[g] = e
        state = context.getState(getEnergy=True, getForces=True)
        out["total"] = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        out["forces"] = state.getForces(asNumpy=True).value_in_unit(
            unit.kilocalories_per_mole / unit.angstrom)
        del context, integrator
        return out

    # Stable covalent groups are the same groups used by the corrected
    # molecule-preserving barostat.  Keeping this construction in the oracle
    # makes the pressure probe independent of OpenMM's barostat internals.
    parent = list(range(n_atoms))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for bond in prmtop.topology.bonds():
        union(bond[0].index, bond[1].index)
    molecule_map = {}
    for index in range(n_atoms):
        molecule_map.setdefault(find(index), []).append(index)
    molecule_groups = list(molecule_map.values())
    masses = np.array(
        [system.getParticleMass(i).value_in_unit(unit.dalton)
         for i in range(n_atoms)], dtype=float
    )

    def scaled_coordinates(coords, factor, molecular):
        scaled = [dict(p) for p in coords]
        if molecular:
            for group in molecule_groups:
                cx = sum(coords[i]["x"] for i in group) / len(group)
                cy = sum(coords[i]["y"] for i in group) / len(group)
                cz = sum(coords[i]["z"] for i in group) / len(group)
                for i in group:
                    scaled[i]["x"] += (factor - 1.0) * cx
                    scaled[i]["y"] += (factor - 1.0) * cy
                    scaled[i]["z"] += (factor - 1.0) * cz
        else:
            for p in scaled:
                p["x"] *= factor
                p["y"] *= factor
                p["z"] *= factor
        return scaled

    def volume_derivative(system, coords, base_energy, molecular=False, h=1e-6):
        """Finite-difference volume derivative under one scaling convention.

        ``molecular=False`` is retained as an atomic-virial diagnostic.  The
        constrained pressure estimator and barostat use ``molecular=True`` so
        rigid waters and covalent groups are never stretched by the probe.
        """
        def scaled_energy(s):
            scaled_coords = scaled_coordinates(coords, s, molecular)
            scaled_box = tuple(mm.Vec3(v.x * s, v.y * s, v.z * s) for v in box)
            integrator = mm.VerletIntegrator(0.001)
            context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
            context.setPositions(to_nm(scaled_coords))
            context.setPeriodicBoxVectors(*scaled_box)
            value = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilocalories_per_mole)
            del context, integrator
            return value

        # Central derivative in ln(s); the smaller perturbation avoids a
        # discontinuous hard-cutoff pair entering only one side of the probe.
        ep = scaled_energy(1.0 + h)
        em = scaled_energy(1.0 - h)
        return -(ep - em) / (math.log(1.0 + h) - math.log(1.0 - h))

    def molecular_pressure(system, coords, velocities):
        """OpenMM-side constrained molecular pressure for a supplied state."""
        box_a3 = float(np.prod([v / 10.0 for v in r["boxAngstrom"]])) * 1000.0
        # Match the production pressure probe's documented 1 +/- 1e-3
        # perturbation.  A much smaller perturbation can straddle a hard
        # cutoff and turn the numerical probe into noise.
        derivative = volume_derivative(system, coords, 0.0, molecular=True, h=1e-3)
        velocity_array = np.array(
            [[v["x"], v["y"], v["z"]] for v in velocities], dtype=float
        )
        kinetic_com = 0.0
        for group in molecule_groups:
            total_mass = float(masses[group].sum())
            velocity = (velocity_array[group] * masses[group, None]).sum(axis=0) / total_mass
            kinetic_com += total_mass * float(np.dot(velocity, velocity)) / (2.0 * 418.4)
        # volume_derivative returns W = -dU/dln(s).  Since dln(V)=3dln(s),
        # the configurational contribution is W/(3V).
        return (2.0 * kinetic_com / (3.0 * box_a3) + derivative / (3.0 * box_a3)) * (4184.0 * 1e25 / 6.02214076e23)

    system_lj, nb_lj = make_system(rigid_water=True)
    zero_charges(system_lj, nb_lj)
    n_constraints = system.getNumConstraints()

    # Targeted statistical reruns keep the same oracle code and settings.
    # Their report intentionally contains only layer 3, not a parity claim.
    if "--nvt-only" not in sys.argv[2:]:
        # ---- Layer 1: single points, per term, rigid + LJ leg ----
        layer1 = {}
        for name, key, ek, fk, ck in [("A", "snapshotA", "energyA", "forcesA", "componentsA"),
                                      ("B", "snapshotB", "energyB", "forcesB", "componentsB")]:
            full = run_points(system, r[key])
            lj = run_points(system_lj, r[key], groups=(3,))
            de = abs(full["total"] - r[ek])
            df = float(abs(full["forces"] - np.asarray(r[fk])).max())
            ours_c = r[ck]
            ours_terms = {
                "bond": ours_c["bonds"],
                "angle": ours_c["angles"],
                "torsion": ours_c["proper_torsions"] + ours_c["improper_torsions"],
                "lj": None,  # filled below
                "electrostatics": None,
            }
            omm_nb = full[3]
            omm_lj = lj[3]
            omm_rf = omm_nb - omm_lj
            ours_terms["lj"] = ours_c["van_der_waals"]
            ours_terms["electrostatics"] = ours_c["electrostatics"]
            term_err = {
                "bond": abs(full[0] - ours_terms["bond"]),
                "angle": abs(full[1] - ours_terms["angle"]),
                "torsion": abs(full[2] - ours_terms["torsion"]),
                "lj": abs(omm_lj - ours_terms["lj"]),
                "electrostatics": abs(omm_rf - ours_terms["electrostatics"]),
            }
            layer1[name] = {
                "openmmTotal": full["total"], "ourTotal": r[ek], "absEnergyError": de,
                "maxForceComponentError": df,
                "openmmTerms": {"bond": full[0], "angle": full[1], "torsion": full[2],
                                "lj": omm_lj, "electrostatics": omm_rf},
                "ourTerms": ours_terms,
                "termErrors": term_err,
            }
            if de > TOL["energy_abs_kcal_mol"]:
                failures.append(f"{name} total energy error {de:.4f}")
            if df > TOL["force_abs_kcal_mol_A"]:
                failures.append(f"{name} force error {df:.4f}")
            for term, err in term_err.items():
                if err > TOL["term_abs_kcal_mol"]:
                    failures.append(f"{name} term {term} error {err:.4f}")
            omm_virial = volume_derivative(system, r[key], full["total"], molecular=False)
            our_virial = r.get("virial" + name)
            if our_virial is not None:
                virial_error = abs(omm_virial - our_virial)
                virial_tol = max(0.5, 1e-4 * abs(omm_virial))
                if virial_error > virial_tol:
                    failures.append(
                        f"{name} virial error {virial_error:.4f} > {virial_tol:.4f}"
                    )
            else:
                virial_error = None
                virial_tol = None
                failures.append(f"{name} missing GlySys analytic virial")
            layer1[name]["virial"] = {
                "openmmVolumeDerivative": omm_virial,
                "ourAnalytic": our_virial,
                "absError": virial_error,
                "tolerance": virial_tol,
            }
            # This is the validated constrained-pressure observable.  It is
            # deliberately separate from the atomic virial above because the
            # latter does not preserve rigid molecular geometry.
            velocity_key = "velocities" + name
            our_pressure = r.get("molecularPressure" + name)
            if our_pressure is not None and velocity_key in r:
                omm_pressure = molecular_pressure(system, r[key], r[velocity_key])
                pressure_error = abs(omm_pressure - our_pressure)
                pressure_tol = max(0.5, 1e-5 * abs(omm_pressure))
                if pressure_error > pressure_tol:
                    failures.append(
                        f"{name} molecular pressure error {pressure_error:.4f} > {pressure_tol:.4f}"
                    )
                layer1[name]["molecularPressure"] = {
                    "openmm": omm_pressure,
                    "ours": our_pressure,
                    "absError": pressure_error,
                    "tolerance": pressure_tol,
                }
            elif name == "A":
                failures.append("A missing molecular pressure fixture fields")
        report["layer1_single_point"] = layer1

        def total_series(system, coords, dt_ps, steps, seed, sample_every=10,
                         initial_velocities=None):
            integrator = mm.VerletIntegrator(dt_ps)
            context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
            context.setPositions(to_nm(coords))
            if initial_velocities is None:
                context.setVelocitiesToTemperature(r["temperatureK"] * unit.kelvin, seed)
            else:
                context.setVelocities(to_nm(initial_velocities))
            if box is not None:
                context.setPeriodicBoxVectors(*box)
            series = []
            for step in range(steps):
                integrator.step(1)
                if (step + 1) % sample_every == 0:
                    state = context.getState(getEnergy=True)
                    pe = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
                    ke = state.getKineticEnergy().value_in_unit(unit.kilocalories_per_mole)
                    series.append(pe + ke)
            del context, integrator
            return series

        def drift_per_atom(series):
            return max(abs(e - series[0]) for e in series) / n_atoms

        # ---- Layer 2: NVE drift, flexible leg, dt convergence ----
        omm_settle = total_series(system, r["snapshotA"], 0.002, 200, 11,
                                  initial_velocities=r.get("velocitiesA"))
        layer2 = {"openmmSettle2fs": drift_per_atom(omm_settle)}
        flex_system, _ = make_system(rigid_water=False)
        omm_flex = total_series(flex_system, r["snapshotA"], 0.001, 200, 11,
                                initial_velocities=r.get("velocitiesA"))
        layer2["openmmFlex1fs"] = drift_per_atom(omm_flex)
        ours_total = r["nveTotalDrift"]
        layer2["ourSettle2fs"] = drift_per_atom(ours_total)
        layer2["ourFlex1fs"] = drift_per_atom(r["flexNve"]["totalDrift"])
        for key in ["ourSettle2fs", "ourFlex1fs", "openmmSettle2fs", "openmmFlex1fs"]:
            short = {"ourSettle2fs": "our settle", "ourFlex1fs": "our flex",
                     "openmmSettle2fs": "openmm settle", "openmmFlex1fs": "openmm flex"}[key]
            if layer2[key] > TOL["nve_drift_per_atom"]:
                failures.append(f"{short} NVE drift {layer2[key]:.4f}/atom")
        conv = []
        for leg in r["nveConvergence"]:
            dt = leg["timestepFs"]
            steps = leg["steps"]
            omm = total_series(system, r["snapshotA"], dt * 0.001, steps, 11,
                               sample_every=1,
                               initial_velocities=r.get("velocitiesA"))
            conv.append({"timestepFs": dt, "steps": steps,
                         "ourDriftPerAtom": leg["maxDriftPerAtom"],
                         "ourEndpointDriftPerAtom": leg["driftPerAtom"],
                         "openmmDriftPerAtom": drift_per_atom(omm)})
        layer2["convergence"] = conv
        # Scaling guard: drift must grow sub-cubically with dt (2nd-order
        # integrators scale ~dt^2; this rules out dt-independent leaks and
        # explosions while allowing chaotic wobble).
        for who in ["ourDriftPerAtom", "openmmDriftPerAtom"]:
            vals = [c[who] for c in conv]
            if min(vals) > 0 and max(vals) / min(vals) > 64.0:
                failures.append(f"NVE convergence {who} spans >64x over 0.5-2fs: {vals}")
        # Cross-code magnitude agreement at production dt.
        c2 = [c for c in conv if c["timestepFs"] == 2.0][0]
        ratio = c2["ourDriftPerAtom"] / max(c2["openmmDriftPerAtom"], 1e-12)
        layer2["driftRatio2fs"] = ratio
        if not (0.1 <= ratio <= 10.0):
            failures.append(f"2fs drift magnitude ratio ours/openmm {ratio:.2f} outside [0.1, 10]")
        report["layer2_nve"] = layer2

    # ---- Layer 3: NVT ensemble means with block standard errors ----
    def block_sem(xs, nblocks=10):
        n = len(xs) // nblocks
        means = [sum(xs[i * n:(i + 1) * n]) / n for i in range(nblocks)]
        m = sum(means) / nblocks
        var = sum((v - m) ** 2 for v in means) / (nblocks - 1)
        return m, math.sqrt(var / nblocks)

    integrator = mm.LangevinMiddleIntegrator(
        r["temperatureK"] * unit.kelvin,
        r["frictionPerPs"] / unit.picosecond,
        r["timestepFs"] * 0.001 * unit.picoseconds,
    )
    # Reproducible independent stream. Seed 0 in OpenMM means a newly chosen
    # seed, which made repeated oracle comparisons change under identical input.
    integrator.setRandomNumberSeed(int(r["nvt"].get("seed", 11)))
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(to_nm(r["snapshotA"]))
    # Start both implementations from the same constrained Maxwell sample.
    # Using an unrelated OpenMM seed here adds a large finite-window kinetic
    # offset and makes a short statistical comparison measure initialization
    # noise rather than thermostat behavior.
    if r.get("velocitiesA") is not None:
        context.setVelocities(to_nm(r["velocitiesA"]))
    else:
        context.setVelocitiesToTemperature(r["temperatureK"] * unit.kelvin, 13)
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    n_equil = int(r["nvt"].get("equilibrationSteps", 500))
    n_prod = int(r["nvt"].get("productionSteps", 2000))
    integrator.step(n_equil)
    # True dynamical DOF: rigid TIP3P removes 3 DOF per water (O-H, O-H, H-H
    # distances all fixed by SETTLE). OpenMM's getNumConstraints() reports 2
    # per water (the listed O-H bonds), so 3N - getNumConstraints() overcounts
    # by one per water (~42 K systematic temperature error here). GlySys
    # SettleWaters likewise constrains all three distances, so both sides use
    # 3N - 3N_water.
    n_rigid_waters = sum(
        1 for res in prmtop.topology.residues() if res.name in ("HOH", "WAT")
    )
    n_solute_h = sum(
        1 for a in topo_atoms if a.element is not None and a.element.symbol == "H"
        and a.residue.name not in ("HOH", "WAT")
    )
    dof = 3 * n_atoms - 3 * n_rigid_waters - n_solute_h
    # LFMiddle stores half-step velocities; GlySys BAOAB stores on-step
    # velocities. Compare the same observable, without perturbing the oracle
    # trajectory: half-kick and RATTLE in a separate analysis context.
    # https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html
    analysis_integrator = mm.VerletIntegrator(r["timestepFs"] * 0.001)
    analysis_context = mm.Context(system, analysis_integrator, mm.Platform.getPlatformByName("Reference"))
    if box is not None:
        analysis_context.setPeriodicBoxVectors(*box)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_atoms)])
    half_dt = r["timestepFs"] * 0.0005
    o_t, o_pe, o_ke, raw_t, raw_ke = [], [], [], [], []
    for step in range(n_prod):
        integrator.step(1)
        if (step + 1) % 10 == 0:
            state = context.getState(getEnergy=True, getPositions=True, getVelocities=True, getForces=True)
            raw = state.getKineticEnergy().value_in_unit(unit.kilocalories_per_mole)
            raw_ke.append(raw); raw_t.append(2.0 * raw / (dof * KB))
            velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            analysis_context.setPositions(state.getPositions())
            analysis_context.setVelocities((velocities + half_dt * forces / masses[:, None]) * unit.nanometer/unit.picosecond)
            analysis_context.applyVelocityConstraints(1e-10)
            on_step = analysis_context.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
            ke = float(0.5 * np.sum(masses[:, None] * on_step**2) / 4.184)
            o_t.append(2.0 * ke / (dof * KB))
            o_pe.append(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
            o_ke.append(ke)
    del context, integrator, analysis_context, analysis_integrator
    n = r["nvt"]
    om, osem = block_sem(o_t)
    opm, opsem = block_sem(o_pe)
    okm, oksem = block_sem(o_ke)
    um, usem = block_sem(n["tempSeries"])
    upm, upsem = block_sem(n["peSeries"])
    ukm, uksem = block_sem(n["keSeries"])
    layer3 = {
        "kineticObservable": "on-step velocity, half-kick plus RATTLE for OpenMM LFMiddle",
        "openmmRawHalfStep": {"meanT": block_sem(raw_t)[0], "meanKE": block_sem(raw_ke)[0]},
        "openmm": {"meanT": om, "semT": osem, "meanPE": opm, "semPE": opsem,
                   "meanKE": okm, "semKE": oksem, "dof": dof,
                   "nConstraints": system.getNumConstraints()},
        "ours": {"meanT": um, "semT": usem, "meanPE": upm, "semPE": upsem,
                 "meanKE": ukm, "semKE": uksem},
    }
    # Stationarity: production halves must agree (catches cold-start
    # transients masquerading as thermostat bias). Diagnostic failure: a
    # drifting run cannot validate any thermostat.
    def half_means(xs):
        h = len(xs) // 2
        return sum(xs[:h]) / h, sum(xs[h:]) / h
    for tag, ours_s, omm_s in [("T", n["tempSeries"], o_t),
                               ("PE", n["peSeries"], o_pe)]:
        oh0, oh1 = half_means(omm_s)
        uh0, uh1 = half_means(ours_s)
        layer3.setdefault("stationarity", {})[tag] = {
            "openmmHalves": [oh0, oh1], "ourHalves": [uh0, uh1]}
        if abs(oh1 - oh0) > max(5.0, 0.05 * abs(oh0)):
            failures.append(f"NVT {tag} openmm production drifting: halves {oh0:.1f} vs {oh1:.1f}")
        if abs(uh1 - uh0) > max(5.0, 0.05 * abs(uh0)):
            failures.append(f"NVT {tag} our production drifting: halves {uh0:.1f} vs {uh1:.1f}")
    report["layer3_nvt"] = layer3
    # Adaptive tolerances: 3x combined SEM with floors covering representation
    # differences (prmtop rounding, RF constant evaluation order) that no
    # sampling statistic can see.
    t_tol = max(TOL["nvt_temperature_K"], 3 * math.hypot(osem, usem))
    e_tol = max(TOL["nvt_energy_kcal_mol"], 3 * math.hypot(opsem, upsem))
    ke_tol = max(2.0, 3 * math.hypot(oksem, uksem))
    layer3["tolerancesUsed"] = {"T": t_tol, "PE": e_tol, "KE": ke_tol}
    if abs(block_sem(raw_t)[0] - r["temperatureK"]) > 15.0:
        failures.append("openmm NVT temperature off target")
    if abs(om - um) > t_tol:
        failures.append(f"NVT temperature mismatch {om:.1f} vs {um:.1f} (tol {t_tol:.1f})")
    if abs(opm - upm) > e_tol:
        failures.append(f"NVT potential mismatch {opm:.1f} vs {upm:.1f} (tol {e_tol:.1f})")
    if abs(okm - ukm) > ke_tol:
        failures.append(f"NVT kinetic mismatch {okm:.1f} vs {ukm:.1f} (tol {ke_tol:.1f})")

print(json.dumps(report, indent=2))
if failures:
    print("FAILURES:", failures, file=sys.stderr)
    sys.exit(1)
print("PBC-RF parity checks passed within statistical bounds")
