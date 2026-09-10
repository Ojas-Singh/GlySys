#!/usr/bin/env python3
"""Independent force evaluation of topology exported by reference_export.
No simulation, minimization or parameter changes. NoCutoff, no constraints.
"""
import json, sys, tempfile
from pathlib import Path
import openmm as mm
from openmm import app, unit

data=json.load(open(sys.argv[1]))
with tempfile.TemporaryDirectory() as folder:
    for name in ['system.prmtop','system.inpcrd']:
        Path(folder,name).write_text(data['files'][name])
    topology=app.AmberPrmtopFile(str(Path(folder,'system.prmtop')))
    results=[]
    for reference in data['evaluations']:
        system=topology.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False)
        if reference['obc2']:
            # Native OpenMM OBC2 uses the same radii/screens supplied by GlySys.
            gb=mm.GBSAOBCForce();gb.setSoluteDielectric(1.0);gb.setSolventDielectric(78.5)
            gb.setSurfaceAreaEnergy(0.00542*unit.kilocalories_per_mole/unit.angstrom**2)
            for charge,radius,screen in data['gbParameters']:
                gb.addParticle(charge,radius*0.1,screen)
            system.addForce(gb)
        names=[]
        for i,force in enumerate(system.getForces()):
            force.setForceGroup(i);names.append(type(force).__name__)
        integrator=mm.VerletIntegrator(0.001)
        context=mm.Context(system,integrator,mm.Platform.getPlatformByName('Reference'))
        context.setPositions([mm.Vec3(p['x']*0.1,p['y']*0.1,p['z']*0.1) for p in data['coordinates']])
        state=context.getState(getEnergy=True,getForces=True)
        energy=state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        forces=state.getForces().value_in_unit(unit.kilocalories_per_mole/unit.angstrom)
        components={name:context.getState(getEnergy=True,groups={i}).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole) for i,name in enumerate(names)}
        expected=sum(reference['components'].values())
        error=max(abs(force[j]+gradient[key]) for force,gradient in zip(forces,reference['gradients']) for j,key in enumerate(['x','y','z']))
        results.append({'obc2':reference['obc2'],'openmmComponents':components,'glysysComponents':reference['components'],'energyError':energy-expected,'maxGradientError':error})
        del context,integrator
print(json.dumps({'schemaVersion':1,'openmmVersion':mm.__version__,'fixture':data['fixture'],'platform':'Reference','results':results},indent=2))
if any(abs(r['energyError'])>1e-3 or r['maxGradientError']>1e-3 for r in results):sys.exit(1)
