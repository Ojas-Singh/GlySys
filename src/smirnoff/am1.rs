//! AM1 semi-empirical molecular orbital calculation (closed shell).
//!
//! Used for AM1-BCC partial charges.  The NDDO integral scheme follows
//! MOPAC (https://github.com/openmopac/mopac, Apache License 2.0, Copyright
//! 2021 Virginia Polytechnic Institute and State University): the AM1
//! parameters below are MOPAC's `parameters_for_AM1_C.F90`; the multipole
//! charge separations and additive terms follow `calpar.F90`, the 22
//! local-frame two-centre integrals `reppd` (mndod.F90), and the core-core
//! repulsion `ccrep.F90`.  Slater-orbital overlaps are evaluated by exact
//! Gauss-Laguerre/Gauss-Legendre quadrature in prolate spheroidal
//! coordinates.

const EV: f64 = 27.211_386_02;
const BOHR: f64 = 0.529_177_210_67;
const EV_TO_KCAL: f64 = 23.060_547_83;

#[derive(Debug, Clone, Copy)]
struct Element {
    z: u8,
    /// Principal quantum number of the valence shell.
    n: u32,
    uss: f64,
    upp: f64,
    betas: f64,
    betap: f64,
    zs: f64,
    zp: f64,
    alp: f64,
    gss: f64,
    gsp: f64,
    gpp: f64,
    gp2: f64,
    hsp: f64,
    eheat: f64,
    gaussians: &'static [(f64, f64, f64)],
}

// N Hsp is 3.14 eV, not an approximation of pi.
#[allow(clippy::approx_constant)]
#[rustfmt::skip]
const ELEMENTS: &[Element] = &[
    Element { z: 1, n: 1, uss: -11.396427, upp: 0.0, betas: -6.173787, betap: 0.0, zs: 1.188078, zp: 0.0, alp: 2.882324, gss: 12.848, gsp: 0.0, gpp: 0.0, gp2: 0.0, hsp: 0.0, eheat: 52.102, gaussians: &[(0.122796, 5.0, 1.2), (0.00509, 5.0, 1.8), (-0.018336, 2.0, 2.1)] },
    Element { z: 6, n: 2, uss: -52.028658, upp: -39.614239, betas: -15.715783, betap: -7.719283, zs: 1.808665, zp: 1.685116, alp: 2.648274, gss: 12.23, gsp: 11.47, gpp: 11.08, gp2: 9.84, hsp: 2.43, eheat: 170.89, gaussians: &[(0.011355, 5.0, 1.6), (0.045924, 5.0, 1.85), (-0.020061, 5.0, 2.05), (-0.00126, 5.0, 2.65)] },
    Element { z: 7, n: 2, uss: -71.86, upp: -57.167581, betas: -20.29911, betap: -18.238666, zs: 2.31541, zp: 2.15794, alp: 2.947286, gss: 13.59, gsp: 12.66, gpp: 12.98, gp2: 11.59, hsp: 3.14, eheat: 113.0, gaussians: &[(0.025251, 5.0, 1.5), (0.028953, 5.0, 2.1), (-0.005806, 2.0, 2.4)] },
    Element { z: 8, n: 2, uss: -97.83, upp: -78.26238, betas: -29.272773, betap: -29.272773, zs: 3.108032, zp: 2.524039, alp: 4.455371, gss: 15.42, gsp: 14.48, gpp: 14.52, gp2: 12.98, hsp: 3.94, eheat: 59.559, gaussians: &[(0.280962, 5.0, 0.847918), (0.08143, 7.0, 1.445071)] },
    Element { z: 9, n: 2, uss: -136.105579, upp: -104.889885, betas: -69.590277, betap: -27.92236, zs: 3.770082, zp: 2.49467, alp: 5.5178, gss: 16.92, gsp: 17.25, gpp: 16.71, gp2: 14.91, hsp: 4.83, eheat: 18.89, gaussians: &[(0.242079, 4.8, 0.93), (0.003607, 4.6, 1.66)] },
    Element { z: 15, n: 3, uss: -42.029863, upp: -34.030709, betas: -6.353764, betap: -6.590709, zs: 1.98128, zp: 1.87515, alp: 2.455322, gss: 11.560005, gsp: 5.237449, gpp: 7.877589, gp2: 7.307648, hsp: 0.779238, eheat: 75.57, gaussians: &[(-0.031827, 6.0, 1.474323), (0.01847, 7.0, 1.779354), (0.03329, 9.0, 3.006576)] },
    Element { z: 16, n: 3, uss: -56.694056, upp: -48.717049, betas: -3.920566, betap: -7.905278, zs: 2.366515, zp: 1.667263, alp: 2.461648, gss: 11.786329, gsp: 8.663127, gpp: 10.039308, gp2: 7.781688, hsp: 2.532137, eheat: 66.4, gaussians: &[(-0.509195, 4.593691, 0.770665), (-0.011863, 5.865731, 1.503313), (0.012334, 13.557336, 2.009173)] },
    Element { z: 17, n: 3, uss: -111.613948, upp: -76.640107, betas: -24.59467, betap: -14.637216, zs: 3.631376, zp: 2.076799, alp: 2.919368, gss: 15.03, gsp: 13.16, gpp: 11.3, gp2: 9.97, hsp: 2.42, eheat: 28.99, gaussians: &[(0.094243, 4.0, 1.3), (0.027168, 4.0, 2.1)] },
    Element { z: 35, n: 4, uss: -104.656063, upp: -74.930052, betas: -19.39988, betap: -8.957195, zs: 3.064133, zp: 2.038333, alp: 2.576546, gss: 15.0364395, gsp: 13.0346824, gpp: 11.2763254, gp2: 9.8544255, hsp: 2.4558683, eheat: 26.74, gaussians: &[(0.066685, 4.0, 1.5), (0.025568, 4.0, 2.3)] },
    Element { z: 53, n: 5, uss: -103.589663, upp: -74.429997, betas: -8.443327, betap: -6.323405, zs: 2.102858, zp: 2.161153, alp: 2.299424, gss: 15.0404486, gsp: 13.056558, gpp: 11.1477837, gp2: 9.9140907, hsp: 2.456382, eheat: 25.517, gaussians: &[(0.004361, 2.3, 1.8), (0.015706, 3.0, 2.24)] },
];

/// Elements with AM1 parameters.
pub fn supported(z: u8) -> bool {
    ELEMENTS.iter().any(|element| element.z == z)
}

/// Element parameters with derived NDDO quantities (atomic units).
#[derive(Debug, Clone, Copy)]
struct Atom {
    element: Element,
    /// Core charge (valence electrons of the neutral atom).
    core: f64,
    orbitals: usize,
    /// Dipole (sp) and quadrupole (pp) charge separations, bohr.
    dd: f64,
    qq: f64,
    /// Klopman-Ohno additive terms expressed as 1/(2 rho), hartree.
    am: f64,
    ad: f64,
    aq: f64,
    eisol: f64,
}

fn derive(element: Element) -> Atom {
    let (s_electrons, p_electrons) = match element.z {
        1 => (1.0, 0.0),
        6 => (2.0, 2.0),
        7 | 15 => (2.0, 3.0),
        8 | 16 => (2.0, 4.0),
        _ => (2.0, 5.0),
    };
    let core = s_electrons + p_electrons;
    if element.z == 1 {
        let am = element.gss / EV;
        return Atom {
            element,
            core,
            orbitals: 1,
            dd: 0.0,
            qq: 0.0,
            am,
            ad: am,
            aq: am,
            eisol: element.uss,
        };
    }
    let n = element.n as f64;
    let (zs, zp) = (element.zs, element.zp);
    let dd = (2.0 * n + 1.0) * (4.0 * zs * zp).powf(n + 0.5)
        / (zs + zp).powf(2.0 * n + 2.0)
        / 3f64.sqrt();
    let qq = ((4.0 * n * n + 6.0 * n + 2.0) / 20.0).sqrt() / zp;
    let hsp = element.hsp.max(1.0e-7);
    let hpp = (0.5 * (element.gpp - element.gp2)).max(0.1);
    // Secant solutions for the additive terms (calpar).
    let mut d1 = (hsp / (EV * dd * dd)).powf(1.0 / 3.0);
    let mut d2 = d1 + 0.04;
    for _ in 0..5 {
        let f = |d: f64| 0.5 * d - 0.5 / (4.0 * dd * dd + 1.0 / (d * d)).sqrt();
        let (h1, h2) = (f(d1), f(d2));
        if (h2 - h1).abs() < 1.0e-25 {
            break;
        }
        let d3 = d1 + (d2 - d1) * (hsp / EV - h1) / (h2 - h1);
        d1 = d2;
        d2 = d3;
    }
    let mut q1 = (16.0 * hpp / (EV * 48.0 * qq.powi(4))).powf(0.2);
    let mut q2 = q1 + 0.04;
    for _ in 0..5 {
        let f = |q: f64| {
            0.25 * q - 0.5 / (4.0 * qq * qq + 1.0 / (q * q)).sqrt()
                + 0.25 / (8.0 * qq * qq + 1.0 / (q * q)).sqrt()
        };
        let (h1, h2) = (f(q1), f(q2));
        if (h2 - h1).abs() < 1.0e-25 {
            break;
        }
        let q3 = q1 + (q2 - q1) * (hpp / EV - h1) / (h2 - h1);
        q1 = q2;
        q2 = q3;
    }
    let k = p_electrons;
    let l = k.min(6.0 - k);
    let gssc = (s_electrons - 1.0f64).max(0.0);
    let gspc = s_electrons * k;
    let gp2c = k * (k - 1.0) / 2.0 + 0.5 * l * (l - 1.0) / 2.0;
    let gppc = -0.5 * l * (l - 1.0) / 2.0;
    let hspc = -k * s_electrons * 0.5;
    let eisol = element.uss * s_electrons
        + element.upp * p_electrons
        + element.gss * gssc
        + element.gpp * gppc
        + element.gsp * gspc
        + element.gp2 * gp2c
        + hsp * hspc;
    Atom {
        element,
        core,
        orbitals: 4,
        dd,
        qq,
        am: element.gss / EV,
        ad: d2,
        aq: q2,
        eisol,
    }
}

/// The 22 unique local-frame two-centre integrals (eV), `reppd` numbering.
/// Local z points from A to B; p-sigma orbitals on both atoms point along +z.
fn local_integrals(a: &Atom, b: &Atom, r_bohr: f64) -> [f64; 22] {
    let mut ri = [0.0; 22];
    let r = r_bohr;
    let (ev1, ev2, ev3, ev4) = (EV / 2.0, EV / 4.0, EV / 8.0, EV / 16.0);
    let pp = 0.5;
    let aee = (pp / a.am + pp / b.am).powi(2);
    let heavy_a = a.orbitals > 1;
    let heavy_b = b.orbitals > 1;
    if !heavy_a && !heavy_b {
        ri[0] = EV / (r * r + aee).sqrt();
        return ri;
    }
    if heavy_a && !heavy_b {
        let da = a.dd;
        let qa = a.qq * 2.0;
        let ade = (pp / a.ad + pp / b.am).powi(2);
        let aqe = (pp / a.aq + pp / b.am).powi(2);
        let s = |x: f64| x.sqrt();
        let ee = EV / s(r * r + aee);
        ri[0] = ee;
        ri[1] = ev1 / s((r + da).powi(2) + ade) - ev1 / s((r - da).powi(2) + ade);
        ri[2] = ee + ev2 / s((r + qa).powi(2) + aqe) + ev2 / s((r - qa).powi(2) + aqe)
            - ev1 / s(r * r + aqe);
        ri[3] = ee + ev1 / s(r * r + aqe + qa * qa) - ev1 / s(r * r + aqe);
        return apply_signs(ri);
    }
    if !heavy_a && heavy_b {
        let db = b.dd;
        let qb = b.qq * 2.0;
        let aed = (pp / a.am + pp / b.ad).powi(2);
        let aeq = (pp / a.am + pp / b.aq).powi(2);
        let s = |x: f64| x.sqrt();
        let ee = EV / s(r * r + aee);
        ri[0] = ee;
        ri[4] = ev1 / s((r - db).powi(2) + aed) - ev1 / s((r + db).powi(2) + aed);
        ri[10] = ee + ev2 / s((r - qb).powi(2) + aeq) + ev2 / s((r + qb).powi(2) + aeq)
            - ev1 / s(r * r + aeq);
        ri[11] = ee + ev1 / s(r * r + aeq + qb * qb) - ev1 / s(r * r + aeq);
        return apply_signs(ri);
    }
    let da = a.dd;
    let db = b.dd;
    let qa = a.qq * 2.0;
    let qb = b.qq * 2.0;
    let ade = (pp / a.ad + pp / b.am).powi(2);
    let aqe = (pp / a.aq + pp / b.am).powi(2);
    let aed = (pp / a.am + pp / b.ad).powi(2);
    let aeq = (pp / a.am + pp / b.aq).powi(2);
    let axx = (pp / a.ad + pp / b.ad).powi(2);
    let adq = (pp / a.ad + pp / b.aq).powi(2);
    let aqd = (pp / a.aq + pp / b.ad).powi(2);
    let aqq = (pp / a.aq + pp / b.aq).powi(2);
    let rsq = r * r;
    let mut arg = [0.0f64; 73];
    arg[1] = rsq + aee;
    arg[2] = (r + da).powi(2) + ade;
    arg[3] = (r - da).powi(2) + ade;
    arg[4] = (r - qa).powi(2) + aqe;
    arg[5] = (r + qa).powi(2) + aqe;
    arg[6] = rsq + aqe;
    arg[7] = arg[6] + qa * qa;
    arg[8] = (r - db).powi(2) + aed;
    arg[9] = (r + db).powi(2) + aed;
    arg[10] = (r - qb).powi(2) + aeq;
    arg[11] = (r + qb).powi(2) + aeq;
    arg[12] = rsq + aeq;
    arg[13] = arg[12] + qb * qb;
    arg[14] = rsq + axx + (da - db).powi(2);
    arg[15] = rsq + axx + (da + db).powi(2);
    arg[16] = (r + da - db).powi(2) + axx;
    arg[17] = (r - da + db).powi(2) + axx;
    arg[18] = (r - da - db).powi(2) + axx;
    arg[19] = (r + da + db).powi(2) + axx;
    arg[20] = (r + da).powi(2) + adq;
    arg[21] = arg[20] + qb * qb;
    arg[22] = (r - da).powi(2) + adq;
    arg[23] = arg[22] + qb * qb;
    arg[24] = (r - db).powi(2) + aqd;
    arg[25] = arg[24] + qa * qa;
    arg[26] = (r + db).powi(2) + aqd;
    arg[27] = arg[26] + qa * qa;
    arg[28] = (r + da - qb).powi(2) + adq;
    arg[29] = (r - da - qb).powi(2) + adq;
    arg[30] = (r + da + qb).powi(2) + adq;
    arg[31] = (r - da + qb).powi(2) + adq;
    arg[32] = (r + qa - db).powi(2) + aqd;
    arg[33] = (r + qa + db).powi(2) + aqd;
    arg[34] = (r - qa - db).powi(2) + aqd;
    arg[35] = (r - qa + db).powi(2) + aqd;
    arg[36] = rsq + aqq;
    arg[37] = arg[36] + (qa - qb).powi(2);
    arg[38] = arg[36] + (qa + qb).powi(2);
    arg[39] = arg[36] + qa * qa;
    arg[40] = arg[36] + qb * qb;
    arg[41] = arg[39] + qb * qb;
    arg[42] = (r - qb).powi(2) + aqq;
    arg[43] = arg[42] + qa * qa;
    arg[44] = (r + qb).powi(2) + aqq;
    arg[45] = arg[44] + qa * qa;
    arg[46] = (r + qa).powi(2) + aqq;
    arg[47] = arg[46] + qb * qb;
    arg[48] = (r - qa).powi(2) + aqq;
    arg[49] = arg[48] + qb * qb;
    arg[50] = (r + qa - qb).powi(2) + aqq;
    arg[51] = (r + qa + qb).powi(2) + aqq;
    arg[52] = (r - qa - qb).powi(2) + aqq;
    arg[53] = (r - qa + qb).powi(2) + aqq;
    let (qa1, qb1) = (a.qq, b.qq);
    let xxx = (da - qb1).powi(2);
    let yyy = (r - qb1).powi(2);
    let zzz = (da + qb1).powi(2);
    let www = (r + qb1).powi(2);
    arg[54] = xxx + yyy + adq;
    arg[55] = xxx + www + adq;
    arg[56] = zzz + yyy + adq;
    arg[57] = zzz + www + adq;
    let xxx = (qa1 - db).powi(2);
    let yyy = (qa1 + db).powi(2);
    let zzz = (r + qa1).powi(2);
    let www = (r - qa1).powi(2);
    arg[58] = zzz + xxx + aqd;
    arg[59] = www + xxx + aqd;
    arg[60] = zzz + yyy + aqd;
    arg[61] = www + yyy + aqd;
    let xxx = (qa1 - qb1).powi(2);
    arg[62] = arg[36] + 2.0 * xxx;
    let yyy = (qa1 + qb1).powi(2);
    arg[63] = arg[36] + 2.0 * yyy;
    arg[64] = arg[36] + 2.0 * (qa1 * qa1 + qb1 * qb1);
    let zzz = (r + qa1 - qb1).powi(2);
    arg[65] = zzz + xxx + aqq;
    arg[66] = zzz + yyy + aqq;
    let zzz = (r + qa1 + qb1).powi(2);
    arg[67] = zzz + xxx + aqq;
    arg[68] = zzz + yyy + aqq;
    let zzz = (r - qa1 - qb1).powi(2);
    arg[69] = zzz + xxx + aqq;
    arg[70] = zzz + yyy + aqq;
    let zzz = (r - qa1 + qb1).powi(2);
    arg[71] = zzz + xxx + aqq;
    arg[72] = zzz + yyy + aqq;
    let mut sqr = [0.0f64; 73];
    for i in 1..73 {
        sqr[i] = arg[i].sqrt();
    }
    let ee = EV / sqr[1];
    let dze = -ev1 / sqr[2] + ev1 / sqr[3];
    let qzze = ev2 / sqr[4] + ev2 / sqr[5] - ev1 / sqr[6];
    let qxxe = ev1 / sqr[7] - ev1 / sqr[6];
    let edz = -ev1 / sqr[8] + ev1 / sqr[9];
    let eqzz = ev2 / sqr[10] + ev2 / sqr[11] - ev1 / sqr[12];
    let eqxx = ev1 / sqr[13] - ev1 / sqr[12];
    let dxdx = ev1 / sqr[14] - ev1 / sqr[15];
    let dzdz = ev2 / sqr[16] + ev2 / sqr[17] - ev2 / sqr[18] - ev2 / sqr[19];
    let dzqxx = ev2 / sqr[20] - ev2 / sqr[21] - ev2 / sqr[22] + ev2 / sqr[23];
    let qxxdz = ev2 / sqr[24] - ev2 / sqr[25] - ev2 / sqr[26] + ev2 / sqr[27];
    let dzqzz = -ev3 / sqr[28] + ev3 / sqr[29] - ev3 / sqr[30] + ev3 / sqr[31] - ev2 / sqr[22]
        + ev2 / sqr[20];
    let qzzdz = -ev3 / sqr[32] + ev3 / sqr[33] - ev3 / sqr[34] + ev3 / sqr[35] + ev2 / sqr[24]
        - ev2 / sqr[26];
    let qxxqxx = ev3 / sqr[37] + ev3 / sqr[38] - ev2 / sqr[39] - ev2 / sqr[40] + ev2 / sqr[36];
    let qxxqyy = ev2 / sqr[41] - ev2 / sqr[39] - ev2 / sqr[40] + ev2 / sqr[36];
    let qxxqzz = ev3 / sqr[43] + ev3 / sqr[45] - ev3 / sqr[42] - ev3 / sqr[44] - ev2 / sqr[39]
        + ev2 / sqr[36];
    let qzzqxx = ev3 / sqr[47] + ev3 / sqr[49] - ev3 / sqr[46] - ev3 / sqr[48] - ev2 / sqr[40]
        + ev2 / sqr[36];
    let qzzqzz = ev4 / sqr[50] + ev4 / sqr[51] + ev4 / sqr[52] + ev4 / sqr[53]
        - ev3 / sqr[48]
        - ev3 / sqr[46]
        - ev3 / sqr[42]
        - ev3 / sqr[44]
        + ev2 / sqr[36];
    let dxqxz = -ev2 / sqr[54] + ev2 / sqr[55] + ev2 / sqr[56] - ev2 / sqr[57];
    let qxzdx = -ev2 / sqr[58] + ev2 / sqr[59] + ev2 / sqr[60] - ev2 / sqr[61];
    let qxzqxz = ev3 / sqr[65] - ev3 / sqr[67] - ev3 / sqr[69] + ev3 / sqr[71] - ev3 / sqr[66]
        + ev3 / sqr[68]
        + ev3 / sqr[70]
        - ev3 / sqr[72];
    ri[0] = ee;
    ri[1] = -dze;
    ri[2] = ee + qzze;
    ri[3] = ee + qxxe;
    ri[4] = -edz;
    ri[5] = dzdz;
    ri[6] = dxdx;
    ri[7] = -edz - qzzdz;
    ri[8] = -edz - qxxdz;
    ri[9] = -qxzdx;
    ri[10] = ee + eqzz;
    ri[11] = ee + eqxx;
    ri[12] = -dze - dzqzz;
    ri[13] = -dze - dzqxx;
    ri[14] = -dxqxz;
    ri[15] = ee + eqzz + qzze + qzzqzz;
    ri[16] = ee + eqzz + qxxe + qxxqzz;
    ri[17] = ee + eqxx + qzze + qzzqxx;
    ri[18] = ee + eqxx + qxxe + qxxqxx;
    ri[19] = qxzqxz;
    ri[20] = ee + eqxx + qxxe + qxxqyy;
    ri[21] = 0.5 * (qxxqxx - qxxqyy);
    apply_signs(ri)
}

/// MOPAC's `nri` orientation signs.
fn apply_signs(mut ri: [f64; 22]) -> [f64; 22] {
    const NRI: [f64; 22] = [
        1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    for (value, sign) in ri.iter_mut().zip(NRI) {
        *value *= sign;
    }
    ri
}

/// Full local tensor L[a][b][c][d] (orbitals s, x, y, z; z toward B).
fn local_tensor(ri: &[f64; 22]) -> [[[[f64; 4]; 4]; 4]; 4] {
    let mut l = [[[[0.0; 4]; 4]; 4]; 4];
    const S: usize = 0;
    const X: usize = 1;
    const Y: usize = 2;
    const Z: usize = 3;
    let mut set = |a: usize, b: usize, c: usize, d: usize, value: f64| {
        for (p, q) in [(a, b), (b, a)] {
            for (r, s) in [(c, d), (d, c)] {
                l[p][q][r][s] = value;
            }
        }
    };
    set(S, S, S, S, ri[0]);
    set(S, Z, S, S, ri[1]);
    set(Z, Z, S, S, ri[2]);
    set(X, X, S, S, ri[3]);
    set(Y, Y, S, S, ri[3]);
    set(S, S, S, Z, ri[4]);
    set(S, Z, S, Z, ri[5]);
    set(S, X, S, X, ri[6]);
    set(S, Y, S, Y, ri[6]);
    set(Z, Z, S, Z, ri[7]);
    set(X, X, S, Z, ri[8]);
    set(Y, Y, S, Z, ri[8]);
    set(X, Z, S, X, ri[9]);
    set(Y, Z, S, Y, ri[9]);
    set(S, S, Z, Z, ri[10]);
    set(S, S, X, X, ri[11]);
    set(S, S, Y, Y, ri[11]);
    set(S, Z, Z, Z, ri[12]);
    set(S, Z, X, X, ri[13]);
    set(S, Z, Y, Y, ri[13]);
    set(S, X, X, Z, ri[14]);
    set(S, Y, Y, Z, ri[14]);
    set(Z, Z, Z, Z, ri[15]);
    set(X, X, Z, Z, ri[16]);
    set(Y, Y, Z, Z, ri[16]);
    set(Z, Z, X, X, ri[17]);
    set(Z, Z, Y, Y, ri[17]);
    set(X, X, X, X, ri[18]);
    set(Y, Y, Y, Y, ri[18]);
    set(X, Z, X, Z, ri[19]);
    set(Y, Z, Y, Z, ri[19]);
    set(X, X, Y, Y, ri[20]);
    set(Y, Y, X, X, ri[20]);
    set(X, Y, X, Y, ri[21]);
    l
}

/// Orthonormal local frame with z along `axis` (unit), rows = local axes.
fn local_frame(axis: [f64; 3]) -> [[f64; 3]; 3] {
    let reference = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let dot = axis[0] * reference[0] + axis[1] * reference[1] + axis[2] * reference[2];
    let mut x = [
        reference[0] - dot * axis[0],
        reference[1] - dot * axis[1],
        reference[2] - dot * axis[2],
    ];
    let norm = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    x = [x[0] / norm, x[1] / norm, x[2] / norm];
    let y = [
        axis[1] * x[2] - axis[2] * x[1],
        axis[2] * x[0] - axis[0] * x[2],
        axis[0] * x[1] - axis[1] * x[0],
    ];
    [x, y, axis]
}

/// T[mu][a]: molecular orbital mu (s, px, py, pz) in local orbitals a.
fn orbital_transform(frame: &[[f64; 3]; 3]) -> [[f64; 4]; 4] {
    let mut t = [[0.0; 4]; 4];
    t[0][0] = 1.0;
    for j in 0..3 {
        for k in 0..3 {
            t[1 + j][1 + k] = frame[k][j];
        }
    }
    t
}

/// Molecular-frame (mu nu | lambda sigma) for orbitals of A and B.
fn rotate_tensor(
    local: &[[[[f64; 4]; 4]; 4]; 4],
    t: &[[f64; 4]; 4],
    na: usize,
    nb: usize,
) -> Vec<f64> {
    // Successive one-index transformations over the 4^4 tensor.
    let mut current = *local;
    for index in 0..4 {
        let mut next = [[[[0.0; 4]; 4]; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    for m in 0..4 {
                        let mut sum = 0.0;
                        for a in 0..4 {
                            let (value, coefficient) = match index {
                                0 => (current[a][j][k][m], t[i][a]),
                                1 => (current[i][a][k][m], t[j][a]),
                                2 => (current[i][j][a][m], t[k][a]),
                                _ => (current[i][j][k][a], t[m][a]),
                            };
                            sum += coefficient * value;
                        }
                        next[i][j][k][m] = sum;
                    }
                }
            }
        }
        current = next;
    }
    let mut out = vec![0.0; na * na * nb * nb];
    for i in 0..na {
        for j in 0..na {
            for k in 0..nb {
                for m in 0..nb {
                    out[((i * na + j) * nb + k) * nb + m] = current[i][j][k][m];
                }
            }
        }
    }
    out
}

/// Gauss-Legendre nodes and weights on [-1, 1].
fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut derivative = 0.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            derivative = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let step = p1 / derivative;
            x -= step;
            if step.abs() < 1.0e-15 {
                break;
            }
        }
        points.push((x, 2.0 / ((1.0 - x * x) * derivative * derivative)));
    }
    points
}

/// Gauss-Laguerre nodes and weights for weight e^-t on [0, inf).
fn gauss_laguerre(n: usize) -> Vec<(f64, f64)> {
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(n);
    let mut z = 0.0;
    for i in 0..n {
        z = match i {
            0 => 3.0 / (1.0 + 2.4 * n as f64),
            1 => z + 15.0 / (1.0 + 2.5 * n as f64),
            _ => {
                let ai = (i - 1) as f64;
                z + ((1.0 + 2.55 * ai) / (1.9 * ai)) * (z - points[i - 2].0)
            }
        };
        let mut derivative = 0.0;
        let mut p_previous = 0.0;
        for _ in 0..200 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for k in 1..=n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * k - 1) as f64 - z) * p2 / k as f64 - (k - 1) as f64 * p3 / k as f64;
            }
            p_previous = p2;
            derivative = n as f64 * (p1 - p2) / z;
            let step = p1 / derivative;
            z -= step;
            if step.abs() < 1.0e-14 * z.abs().max(1.0) {
                break;
            }
        }
        let weight = -1.0 / (derivative * n as f64 * p_previous);
        points.push((z, weight));
    }
    points
}

struct Quadrature {
    laguerre: Vec<(f64, f64)>,
    legendre: Vec<(f64, f64)>,
}

impl Quadrature {
    fn new() -> Self {
        Self {
            laguerre: gauss_laguerre(24),
            legendre: gauss_legendre(40),
        }
    }
}

fn factorial(n: u32) -> f64 {
    (1..=n).map(f64::from).product()
}

/// Local-frame overlaps [ss, s(A)-sigma(B), sigma(A)-s(B), sigma-sigma, pi-pi].
fn local_overlaps(a: &Atom, b: &Atom, r_bohr: f64, quadrature: &Quadrature) -> [f64; 5] {
    let mut result = [0.0; 5];
    let na = a.element.n as i32;
    let nb = b.element.n as i32;
    let norm =
        |n: i32, zeta: f64| (2.0 * zeta).powf(n as f64 + 0.5) / factorial(2 * n as u32).sqrt();
    let half = r_bohr / 2.0;
    // (zeta_a, zeta_b, kind): kind 0 ss, 1 s-sigma, 2 sigma-s, 3 sigma-sigma, 4 pi-pi
    let combos: [(f64, f64, usize); 5] = [
        (a.element.zs, b.element.zs, 0),
        (a.element.zs, b.element.zp, 1),
        (a.element.zp, b.element.zs, 2),
        (a.element.zp, b.element.zp, 3),
        (a.element.zp, b.element.zp, 4),
    ];
    for (zeta_a, zeta_b, kind) in combos {
        let needs_p_a = matches!(kind, 2..=4);
        let needs_p_b = matches!(kind, 1 | 3 | 4);
        if (needs_p_a && a.orbitals == 1) || (needs_p_b && b.orbitals == 1) {
            continue;
        }
        let p = (zeta_a + zeta_b) * half;
        let q = (zeta_a - zeta_b) * half;
        let prefactor = norm(na, zeta_a) * norm(nb, zeta_b) * half.powi(3);
        let mut sum = 0.0;
        for &(t, wt) in &quadrature.laguerre {
            let xi = 1.0 + t / p;
            for &(eta, we) in &quadrature.legendre {
                let ra = half * (xi + eta);
                let rb = half * (xi - eta);
                let radial = ra.powi(na - 1) * rb.powi(nb - 1) * (-q * eta).exp();
                let cos_a = (1.0 + xi * eta) / (xi + eta);
                let cos_b = (xi * eta - 1.0) / (xi - eta);
                let angular = match kind {
                    0 => 0.5,
                    1 => 0.5 * 3f64.sqrt() * cos_b,
                    2 => 0.5 * 3f64.sqrt() * cos_a,
                    3 => 1.5 * cos_a * cos_b,
                    _ => {
                        let sin2 = ((xi * xi - 1.0) * (1.0 - eta * eta)).max(0.0);
                        0.75 * sin2 / ((xi + eta) * (xi - eta))
                    }
                };
                sum += wt * we * (xi * xi - eta * eta) * radial * angular;
            }
        }
        // d(xi) = dt / p and the e^-p factor removed by the substitution.
        result[kind] = prefactor * sum * (-p).exp() / p;
    }
    result
}

/// Result of an AM1 calculation.
#[derive(Debug, Clone)]
pub struct Am1Result {
    /// Mulliken (ZDO) partial charges, e.
    pub charges: Vec<f64>,
    /// Heat of formation, kcal/mol.
    pub heat_of_formation: f64,
    /// SCF iterations used.
    #[allow(dead_code)]
    pub iterations: usize,
}

/// Closed-shell AM1 at the given geometry (Å).
pub fn am1(elements: &[u8], positions: &[[f64; 3]], charge: i32) -> Result<Am1Result, String> {
    let atoms = elements
        .iter()
        .map(|&z| {
            ELEMENTS
                .iter()
                .find(|element| element.z == z)
                .copied()
                .map(derive)
                .ok_or_else(|| format!("AM1 has no parameters for element {z}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let count = atoms.len();
    let mut first = Vec::with_capacity(count);
    let mut total_orbitals = 0;
    for atom in &atoms {
        first.push(total_orbitals);
        total_orbitals += atom.orbitals;
    }
    let electrons = atoms.iter().map(|atom| atom.core).sum::<f64>() - charge as f64;
    if electrons < 0.0 || electrons.fract() != 0.0 || electrons as i64 % 2 != 0 {
        return Err(format!(
            "AM1 needs a closed-shell molecule; {electrons} electrons"
        ));
    }
    let occupied = (electrons as usize) / 2;
    if occupied > total_orbitals {
        return Err("too many electrons for the AM1 basis".into());
    }
    let n = total_orbitals;
    let quadrature = Quadrature::new();

    // Core Hamiltonian and two-centre integrals.
    let mut h = vec![0.0; n * n];
    let mut core_energy = 0.0;
    let mut pairs = Vec::new();
    for (i, atom) in atoms.iter().enumerate() {
        let fi = first[i];
        h[fi * n + fi] = atom.element.uss;
        for k in 1..atom.orbitals {
            h[(fi + k) * n + fi + k] = atom.element.upp;
        }
    }
    for i in 0..count {
        for j in 0..i {
            let (a, b) = (&atoms[i], &atoms[j]);
            let delta = [
                positions[j][0] - positions[i][0],
                positions[j][1] - positions[i][1],
                positions[j][2] - positions[i][2],
            ];
            let r = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            if r < 0.1 {
                return Err("atoms overlap".into());
            }
            let axis = [delta[0] / r, delta[1] / r, delta[2] / r];
            let frame = local_frame(axis);
            let t = orbital_transform(&frame);
            let ri = local_integrals(a, b, r / BOHR);
            let tensor = rotate_tensor(&local_tensor(&ri), &t, a.orbitals, b.orbitals);
            let (na, nb) = (a.orbitals, b.orbitals);
            let (fi, fj) = (first[i], first[j]);
            // Electron-core attraction: -Z_B (mu nu | s s)_B and vice versa.
            for mu in 0..na {
                for nu in 0..na {
                    h[(fi + mu) * n + fi + nu] -= b.core * tensor[((mu * na + nu) * nb) * nb];
                }
            }
            for lambda in 0..nb {
                for sigma in 0..nb {
                    h[(fj + lambda) * n + fj + sigma] -= a.core * tensor[lambda * nb + sigma];
                }
            }
            // Resonance integrals from overlaps.
            let s = local_overlaps(a, b, r / BOHR, &quadrature);
            let mut local_s = [[0.0; 4]; 4];
            local_s[0][0] = s[0];
            local_s[0][3] = s[1];
            local_s[3][0] = s[2];
            local_s[3][3] = s[3];
            local_s[1][1] = s[4];
            local_s[2][2] = s[4];
            for mu in 0..na {
                for lambda in 0..nb {
                    let mut overlap = 0.0;
                    for p in 0..4 {
                        for q in 0..4 {
                            overlap += t[mu][p] * t[lambda][q] * local_s[p][q];
                        }
                    }
                    let beta_mu = if mu == 0 {
                        a.element.betas
                    } else {
                        a.element.betap
                    };
                    let beta_lambda = if lambda == 0 {
                        b.element.betas
                    } else {
                        b.element.betap
                    };
                    let value = 0.5 * (beta_mu + beta_lambda) * overlap;
                    h[(fi + mu) * n + fj + lambda] = value;
                    h[(fj + lambda) * n + fi + mu] = value;
                }
            }
            core_energy += core_core(a, b, r, ri[0]);
            pairs.push((i, j, tensor));
        }
    }

    // SCF.
    let mut density = vec![0.0; n * n];
    let scale = electrons / atoms.iter().map(|atom| atom.core).sum::<f64>();
    for (i, atom) in atoms.iter().enumerate() {
        for k in 0..atom.orbitals {
            let fi = first[i] + k;
            density[fi * n + fi] = scale * atom.core / atom.orbitals as f64;
        }
    }
    // Pulay DIIS from the second iteration; if that stalls, restart from
    // the current density with a virtual-orbital level shift.
    let mut iterations = 0;
    let mut electronic = 0.0;
    let mut converged = false;
    for level_shift in [0.0, 4.0, 12.0] {
        let mut energy_previous = f64::INFINITY;
        let mut diis_focks: Vec<Vec<f64>> = Vec::new();
        let mut diis_errors: Vec<Vec<f64>> = Vec::new();
        let mut best_error = f64::INFINITY;
        let mut stalled = 0;
        for _ in 0..250 {
            iterations += 1;
            let fock = build_fock(&h, &density, &atoms, &first, &pairs, n);
            electronic = 0.5
                * (0..n * n)
                    .map(|k| density[k] * (h[k] + fock[k]))
                    .sum::<f64>();
            let error = commutator(&fock, &density, n);
            let error_norm = error.iter().map(|e| e.abs()).fold(0.0, f64::max);
            if (electronic - energy_previous).abs() < 1.0e-8 && error_norm < 1.0e-5 {
                converged = true;
                break;
            }
            if error_norm < best_error * 0.999 {
                best_error = error_norm;
                stalled = 0;
            } else {
                stalled += 1;
                if stalled > 40 {
                    break;
                }
            }
            energy_previous = electronic;
            diis_focks.push(fock.clone());
            diis_errors.push(error);
            if diis_focks.len() > 10 {
                diis_focks.remove(0);
                diis_errors.remove(0);
            }
            let mut extrapolated = if diis_focks.len() >= 2 {
                diis(&diis_focks, &diis_errors).unwrap_or(fock)
            } else {
                fock
            };
            if level_shift > 0.0 {
                // F + s (1 - P/2): raises virtual orbitals by s eV.
                for mu in 0..n {
                    for nu in 0..n {
                        let identity = if mu == nu { 1.0 } else { 0.0 };
                        extrapolated[mu * n + nu] +=
                            level_shift * (identity - 0.5 * density[mu * n + nu]);
                    }
                }
            }
            let (_, vectors) = symmetric_eigen(&extrapolated, n);
            for mu in 0..n {
                for nu in 0..=mu {
                    let mut sum = 0.0;
                    for k in 0..occupied {
                        sum += vectors[mu * n + k] * vectors[nu * n + k];
                    }
                    density[mu * n + nu] = 2.0 * sum;
                    density[nu * n + mu] = 2.0 * sum;
                }
            }
        }
        if converged {
            break;
        }
    }
    if !converged {
        return Err("AM1 SCF did not converge".into());
    }
    let charges = atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| {
            let population: f64 = (0..atom.orbitals)
                .map(|k| density[(first[i] + k) * n + first[i] + k])
                .sum();
            atom.core - population
        })
        .collect();
    let isolated = atoms.iter().map(|atom| atom.eisol).sum::<f64>();
    let atomic_heats = atoms.iter().map(|atom| atom.element.eheat).sum::<f64>();
    Ok(Am1Result {
        charges,
        heat_of_formation: (electronic + core_energy - isolated) * EV_TO_KCAL + atomic_heats,
        iterations,
    })
}

/// AM1 core-core repulsion (eV) for atoms at distance `r` Å; `gamma` = (ss|ss).
fn core_core(a: &Atom, b: &Atom, r: f64, gamma: f64) -> f64 {
    let enuc = a.core * b.core * gamma;
    let ea = (-a.element.alp * r).exp();
    let eb = (-b.element.alp * r).exp();
    let mut scale = ea + eb;
    let pair = a.element.z as u32 + b.element.z as u32;
    if pair == 8 || pair == 9 {
        // N-H and O-H: MNDO's R*exp(-alpha R) term for the heavy atom.
        if matches!(a.element.z, 7 | 8) {
            scale += (r - 1.0) * ea;
        }
        if matches!(b.element.z, 7 | 8) {
            scale += (r - 1.0) * eb;
        }
    }
    let mut energy = enuc + (scale * enuc).abs();
    for atom in [a, b] {
        for &(k, l, m) in atom.element.gaussians {
            let x = l * (r - m).powi(2);
            if x <= 25.0 {
                energy += a.core * b.core / r * k * (-x).exp();
            }
        }
    }
    energy
}

#[allow(clippy::type_complexity)]
fn build_fock(
    h: &[f64],
    density: &[f64],
    atoms: &[Atom],
    first: &[usize],
    pairs: &[(usize, usize, Vec<f64>)],
    n: usize,
) -> Vec<f64> {
    let mut fock = h.to_vec();
    // One-centre terms.
    for (index, atom) in atoms.iter().enumerate() {
        let f0 = first[index];
        let e = &atom.element;
        let hpp = 0.5 * (e.gpp - e.gp2);
        let one_centre = |mu: usize, nu: usize, lambda: usize, sigma: usize| -> f64 {
            // (mu nu | lambda sigma) on one atom, orbitals 0 = s, 1..3 = p.
            let (a, b, c, d) = (mu, nu, lambda, sigma);
            if a == b && c == d {
                match (a == 0, c == 0) {
                    (true, true) => e.gss,
                    (true, false) | (false, true) => e.gsp,
                    (false, false) if a == c => e.gpp,
                    _ => e.gp2,
                }
            } else if (a == c && b == d) || (a == d && b == c) {
                if a == 0 || b == 0 { e.hsp } else { hpp }
            } else {
                0.0
            }
        };
        for mu in 0..atom.orbitals {
            for nu in 0..atom.orbitals {
                let mut value = 0.0;
                for lambda in 0..atom.orbitals {
                    for sigma in 0..atom.orbitals {
                        let p = density[(f0 + lambda) * n + f0 + sigma];
                        if p == 0.0 {
                            continue;
                        }
                        value += p
                            * (one_centre(mu, nu, lambda, sigma)
                                - 0.5 * one_centre(mu, lambda, nu, sigma));
                    }
                }
                fock[(f0 + mu) * n + f0 + nu] += value;
            }
        }
    }
    // Two-centre Coulomb and exchange.
    for (i, j, tensor) in pairs {
        let (a, b) = (&atoms[*i], &atoms[*j]);
        let (na, nb) = (a.orbitals, b.orbitals);
        let (fi, fj) = (first[*i], first[*j]);
        let at = |mu: usize, nu: usize, lambda: usize, sigma: usize| {
            tensor[((mu * na + nu) * nb + lambda) * nb + sigma]
        };
        for mu in 0..na {
            for nu in 0..na {
                let mut coulomb = 0.0;
                for lambda in 0..nb {
                    for sigma in 0..nb {
                        coulomb +=
                            density[(fj + lambda) * n + fj + sigma] * at(mu, nu, lambda, sigma);
                    }
                }
                fock[(fi + mu) * n + fi + nu] += coulomb;
            }
        }
        for lambda in 0..nb {
            for sigma in 0..nb {
                let mut coulomb = 0.0;
                for mu in 0..na {
                    for nu in 0..na {
                        coulomb += density[(fi + mu) * n + fi + nu] * at(mu, nu, lambda, sigma);
                    }
                }
                fock[(fj + lambda) * n + fj + sigma] += coulomb;
            }
        }
        for mu in 0..na {
            for lambda in 0..nb {
                let mut exchange = 0.0;
                for nu in 0..na {
                    for sigma in 0..nb {
                        exchange += density[(fi + nu) * n + fj + sigma] * at(mu, nu, lambda, sigma);
                    }
                }
                fock[(fi + mu) * n + fj + lambda] -= 0.5 * exchange;
                fock[(fj + lambda) * n + fi + mu] -= 0.5 * exchange;
            }
        }
    }
    fock
}

fn commutator(fock: &[f64], density: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += fock[i * n + k] * density[k * n + j] - density[i * n + k] * fock[k * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    result
}

/// Pulay DIIS extrapolation.
fn diis(focks: &[Vec<f64>], errors: &[Vec<f64>]) -> Option<Vec<f64>> {
    let m = focks.len();
    let size = m + 1;
    let mut b = vec![0.0; size * size];
    for i in 0..m {
        for j in 0..m {
            b[i * size + j] = errors[i].iter().zip(&errors[j]).map(|(x, y)| x * y).sum();
        }
        b[i * size + m] = -1.0;
        b[m * size + i] = -1.0;
    }
    let mut rhs = vec![0.0; size];
    rhs[m] = -1.0;
    let coefficients = solve(&mut b, &mut rhs, size)?;
    let mut fock = vec![0.0; focks[0].len()];
    for (coefficient, matrix) in coefficients.iter().take(m).zip(focks) {
        for (value, element) in fock.iter_mut().zip(matrix) {
            *value += coefficient * element;
        }
    }
    Some(fock)
}

/// Gaussian elimination with partial pivoting.
fn solve(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&x, &y| a[x * n + column].abs().total_cmp(&a[y * n + column].abs()))?;
        if a[pivot * n + column].abs() < 1.0e-14 {
            return None;
        }
        if pivot != column {
            for k in 0..n {
                a.swap(pivot * n + k, column * n + k);
            }
            b.swap(pivot, column);
        }
        for row in column + 1..n {
            let factor = a[row * n + column] / a[column * n + column];
            for k in column..n {
                a[row * n + k] -= factor * a[column * n + k];
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in row + 1..n {
            sum -= a[row * n + k] * x[k];
        }
        x[row] = sum / a[row * n + row];
    }
    Some(x)
}

/// Eigen-decomposition of a symmetric matrix (Householder + implicit QL).
/// Returns ascending eigenvalues and column eigenvectors (row-major n x n).
pub(crate) fn symmetric_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut z = matrix.to_vec();
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    // tred2
    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0;
        if l > 0 {
            let scale: f64 = (0..=l).map(|k| z[i * n + k].abs()).sum();
            if scale == 0.0 {
                e[i] = z[i * n + l];
            } else {
                for k in 0..=l {
                    z[i * n + k] /= scale;
                    h += z[i * n + k] * z[i * n + k];
                }
                let f = z[i * n + l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                z[i * n + l] = f - g;
                let mut f = 0.0;
                for j in 0..=l {
                    z[j * n + i] = z[i * n + j] / h;
                    let mut g = 0.0;
                    for k in 0..=j {
                        g += z[j * n + k] * z[i * n + k];
                    }
                    for k in j + 1..=l {
                        g += z[k * n + j] * z[i * n + k];
                    }
                    e[j] = g / h;
                    f += e[j] * z[i * n + j];
                }
                let hh = f / (h + h);
                for j in 0..=l {
                    let f = z[i * n + j];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 0..=j {
                        z[j * n + k] -= f * e[k] + g * z[i * n + k];
                    }
                }
            }
        } else {
            e[i] = z[i * n + l];
        }
        d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
    for i in 0..n {
        if d[i] != 0.0 {
            for j in 0..i {
                let mut g = 0.0;
                for k in 0..i {
                    g += z[i * n + k] * z[k * n + j];
                }
                for k in 0..i {
                    z[k * n + j] -= g * z[k * n + i];
                }
            }
        }
        d[i] = z[i * n + i];
        z[i * n + i] = 1.0;
        for j in 0..i {
            z[j * n + i] = 0.0;
            z[i * n + j] = 0.0;
        }
    }
    // tql2
    for i in 1..n {
        e[i - 1] = e[i];
    }
    if n > 0 {
        e[n - 1] = 0.0;
    }
    for l in 0..n {
        let mut iterations = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iterations += 1;
            if iterations > 60 {
                break;
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r } else { -r });
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut i = m;
            let mut underflow = false;
            while i > l {
                i -= 1;
                let mut f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                for k in 0..n {
                    f = z[k * n + i + 1];
                    z[k * n + i + 1] = s * z[k * n + i] + c * f;
                    z[k * n + i] = c * z[k * n + i] - s * f;
                }
            }
            if underflow {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    // Sort ascending.
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
    let values = order.iter().map(|&k| d[k]).collect();
    let mut vectors = vec![0.0; n * n];
    for (column, &k) in order.iter().enumerate() {
        for row in 0..n {
            vectors[row * n + column] = z[row * n + k];
        }
    }
    (values, vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_solver_diagonalizes() {
        let m = [4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 1.0];
        let (values, vectors) = symmetric_eigen(&m, 3);
        for k in 0..3 {
            for i in 0..3 {
                let av: f64 = (0..3).map(|j| m[i * 3 + j] * vectors[j * 3 + k]).sum();
                assert!((av - values[k] * vectors[i * 3 + k]).abs() < 1e-10);
            }
        }
        assert!(values[0] <= values[1] && values[1] <= values[2]);
    }

    #[test]
    fn quadrature_reproduces_hydrogen_overlap() {
        // 1s-1s overlap S = e^{-p}(1 + p + p^2/3), p = zeta R.
        let h = derive(ELEMENTS[0]);
        let r = 1.4;
        let s = local_overlaps(&h, &h, r, &Quadrature::new());
        let p = h.element.zs * r;
        let exact = (-p).exp() * (1.0 + p + p * p / 3.0);
        assert!((s[0] - exact).abs() < 1e-12, "{} vs {exact}", s[0]);
    }

    #[test]
    fn local_integral_signs_follow_point_charge_model() {
        // (s pz_A | s s_B): dipole on A with +1/2 toward B.
        let c = derive(ELEMENTS[1]);
        let h = derive(ELEMENTS[0]);
        let r = 2.0;
        let ri = local_integrals(&c, &h, r);
        let ade = (0.5 / c.ad + 0.5 / h.am).powi(2);
        let toward = 0.5 / ((r - c.dd).powi(2) + ade).sqrt();
        let away = 0.5 / ((r + c.dd).powi(2) + ade).sqrt();
        assert!((ri[1] - EV * (toward - away)).abs() < 1e-10);
    }

    #[test]
    fn water_charges_and_heat_of_formation() {
        // AM1 water: heat of formation about -59.2 kcal/mol, q(O) about -0.38.
        let elements = [8, 1, 1];
        let positions = [[0.0, 0.0, 0.0], [0.9614, 0.0, 0.0], [-0.2511, 0.9281, 0.0]];
        let result = am1(&elements, &positions, 0).unwrap();
        assert!((result.charges.iter().sum::<f64>()).abs() < 1e-8);
        assert!(
            result.charges[0] < -0.3 && result.charges[0] > -0.45,
            "{:?}",
            result.charges
        );
        assert!(
            (result.heat_of_formation + 59.2).abs() < 1.5,
            "{}",
            result.heat_of_formation
        );
    }
}
