#!/usr/bin/env python3
"""Independent BAOAB trajectory check using exported identical Amber chemistry."""
import json,sys,tempfile,math
from pathlib import Path
import openmm as mm
from openmm import app,unit
r=json.load(open(sys.argv[1]));results=[]
with tempfile.TemporaryDirectory() as folder:
    Path(folder,'system.prmtop').write_text(r['files']['system.prmtop'])
    amber=app.AmberPrmtopFile(str(Path(folder,'system.prmtop')))
    for case in r['cases']:
        initial=case['initial'];protocol=initial['protocol'];n=len(initial['coordinates'])
        system=amber.createSystem(nonbondedMethod=app.NoCutoff,constraints=None,rigidWater=False,removeCMMotion=False)
        gb=mm.GBSAOBCForce();gb.setSoluteDielectric(1.0);gb.setSolventDielectric(78.5);gb.setSurfaceAreaEnergy(0.00542*unit.kilocalories_per_mole/unit.angstrom**2)
        for q,radius,screen in r['gbParameters']:gb.addParticle(q,radius*0.1,screen)
        system.addForce(gb)
        dt=protocol['timestepFs']*0.001;decay=math.exp(-protocol['frictionPerPs']*dt)
        integrator=mm.CustomIntegrator(dt)
        integrator.addGlobalVariable('decay',decay);integrator.addGlobalVariable('variance',(1-decay*decay)*0.00831446261815324*protocol['temperatureK'])
        integrator.addPerDofVariable('eta',0)
        integrator.addComputePerDof('v','v+0.5*dt*f/m');integrator.addComputePerDof('x','x+0.5*dt*v')
        integrator.addComputePerDof('v','decay*v+sqrt(variance/m)*eta');integrator.addComputePerDof('x','x+0.5*dt*v');integrator.addComputePerDof('v','v+0.5*dt*f/m')
        context=mm.Context(system,integrator,mm.Platform.getPlatformByName('Reference'))
        convert=lambda ps:[mm.Vec3(p['x']*0.1,p['y']*0.1,p['z']*0.1) for p in ps]
        context.setPositions(convert(initial['coordinates']));context.setVelocities(convert(initial['velocities']))
        for step in range(20):
            integrator.setPerDofVariableByName('eta',[mm.Vec3(*p[:3]) for p in case['noise'][step*n:(step+1)*n]])
            integrator.step(1)
        state=context.getState(getPositions=True,getVelocities=True,getEnergy=True)
        pos=state.getPositions().value_in_unit(unit.angstrom);vel=state.getVelocities().value_in_unit(unit.angstrom/unit.picosecond)
        coordinate_error=max(abs(p[i]-ref[k]) for p,ref in zip(pos,case['final']['coordinates']) for i,k in enumerate(['x','y','z']))
        velocity_error=max(abs(p[i]-ref[k]) for p,ref in zip(vel,case['final']['velocities']) for i,k in enumerate(['x','y','z']))
        results.append({'frictionPerPs':protocol['frictionPerPs'],'coordinateErrorAngstrom':coordinate_error,'velocityErrorAngstromPerPs':velocity_error,'potentialEnergyError':state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)-case['final']['potentialEnergy']})
        del context,integrator
print(json.dumps({'schemaVersion':1,'openmmVersion':mm.__version__,'platform':'Reference','results':results},indent=2))
if any(r['coordinateErrorAngstrom']>1e-5 or r['velocityErrorAngstromPerPs']>1e-3 or abs(r['potentialEnergyError'])>1e-3 for r in results):sys.exit(1)
