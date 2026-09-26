struct Step { size:vec4<u32>, values:vec4<f32> }
@group(0) @binding(0) var<uniform> step:Step;
@group(0) @binding(1) var<storage,read_write> coordinates:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> velocities:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> forces:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> noise:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> invalid:atomic<u32>;
struct ConstraintGroup { start:u32,count:u32,padding:vec2<u32> }
struct Constraint { atoms:vec4<u32>,parameters:vec4<f32> }
@group(0) @binding(6) var<storage,read> constraint_groups:array<ConstraintGroup>;
@group(0) @binding(7) var<storage,read> constraints:array<Constraint>;
@group(0) @binding(8) var<storage,read_write> trial_coordinates:array<vec4<f32>>;
@compute @workgroup_size(64)
fn before_force(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=step.size.x){return;}
 let dt=step.values.x;let decay=step.values.y;let invmass=velocities[i].w;
 var v=velocities[i].xyz-0.5*dt*418.4*invmass*forces[3u*step.size.x+i].xyz;
 var movement=0.5*dt*v;
 var random = noise[step.size.y+i].xyz;
 if(step.size.z == 1u) {
   let draw = rng_normal3(bitcast<u32>(noise[i].w));
   noise[i] = draw;
   random = draw.xyz;
 }
 v=decay*v+sqrt(step.values.z*invmass)*random;
 movement+=0.5*dt*v;
 if(dot(movement,movement)>1.){atomicStore(&invalid,1u);}
 coordinates[i]=vec4<f32>(coordinates[i].xyz+movement,1.);
 velocities[i]=vec4<f32>(v,invmass);
}
@compute @workgroup_size(64)
fn after_force(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=step.size.x){return;}
 velocities[i]=vec4<f32>(velocities[i].xyz-0.5*step.values.x*418.4*velocities[i].w*forces[3u*step.size.x+i].xyz,velocities[i].w);
}

// OpenMM LF-middle ordering for constrained NVT dynamics. Constraint groups
// are atom-disjoint X-H stars, so each invocation owns all writes in a group.
@compute @workgroup_size(64)
fn lf_kick_velocity_projection(@builtin(global_invocation_id) id:vec3<u32>){
 let group_index=id.x;if(group_index>=step.size.w){return;}
 let group=constraint_groups[group_index];let parent=group.padding.x;
 let parent_invmass=velocities[parent].w;
 velocities[parent]=vec4<f32>(velocities[parent].xyz-step.values.x*418.4*parent_invmass*forces[3u*step.size.x+parent].xyz,parent_invmass);
 for(var local=0u;local<group.count;local++){let atom=constraints[group.start+local].atoms.y;let invmass=velocities[atom].w;velocities[atom]=vec4<f32>(velocities[atom].xyz-step.values.x*418.4*invmass*forces[3u*step.size.x+atom].xyz,invmass);}
 if(group.count==0u){return;}
 var converged=false;
 for(var iteration=0u;iteration<200u;iteration++){
  var max_residual=0.0;
  for(var local=0u;local<group.count;local++){
   let c=constraints[group.start+local];let a=c.atoms.x;let b=c.atoms.y;
   let delta=coordinates[a].xyz-coordinates[b].xyz;
   let va=velocities[a].xyz;let vb=velocities[b].xyz;
   let residual=dot(delta,va-vb);
   max_residual=max(max_residual,abs(residual)/max(length(delta),1e-8));
   let denominator=dot(delta,delta)*(c.parameters.y+c.parameters.z);
   if(denominator>1e-20){
    let lambda=residual/denominator;
    velocities[a]=vec4<f32>(va-lambda*c.parameters.y*delta,velocities[a].w);
    velocities[b]=vec4<f32>(vb+lambda*c.parameters.z*delta,velocities[b].w);
   }else{atomicOr(&invalid,1u);}
  }
  if(max_residual<=1e-5){converged=true;break;}
 }
 if(!converged){atomicOr(&invalid,1u);}
}
@compute @workgroup_size(64)
fn lf_first_half_drift(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=step.size.x){return;}
 coordinates[i]=vec4<f32>(coordinates[i].xyz+0.5*step.values.x*velocities[i].xyz,1.0);
}
@compute @workgroup_size(64)
fn lf_drift_thermostat_snapshot(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=step.size.x){return;}
 let dt=step.values.x;let invmass=velocities[i].w;let old_position=coordinates[i].xyz;let old_velocity=velocities[i].xyz;
 trial_coordinates[step.size.x+i]=vec4<f32>(old_position,1.0);
 let half_position=old_position+0.5*dt*old_velocity;
 let random=rng_normal3(bitcast<u32>(noise[i].w));noise[i]=random;
 let new_velocity=step.values.y*old_velocity+sqrt(step.values.z*invmass)*random.xyz;
 let trial_position=half_position+0.5*dt*new_velocity;
 if(dot(trial_position-old_position,trial_position-old_position)>1.0){atomicStore(&invalid,1u);}
 velocities[i]=vec4<f32>(new_velocity,invmass);coordinates[i]=vec4<f32>(trial_position,1.0);trial_coordinates[i]=vec4<f32>(trial_position,1.0);
}
@compute @workgroup_size(64)
fn lf_thermostat_second_half(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=step.size.x){return;}
 let invmass=velocities[i].w;let random=rng_normal3(bitcast<u32>(noise[i].w));noise[i]=random;
 let v=step.values.y*velocities[i].xyz+sqrt(step.values.z*invmass)*random.xyz;
 velocities[i]=vec4<f32>(v,invmass);
 coordinates[i]=vec4<f32>(coordinates[i].xyz+0.5*step.values.x*v,1.0);
}
@compute @workgroup_size(64)
fn lf_save_trial(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=step.size.x){return;}trial_coordinates[i]=coordinates[i];
}
@compute @workgroup_size(64)
fn lf_position_projection(@builtin(global_invocation_id) id:vec3<u32>){
 let group_index=id.x;if(group_index>=step.size.w){return;}
 let group=constraint_groups[group_index];
 if(group.count==0u){let atom=group.padding.x;let v=velocities[atom].xyz+(coordinates[atom].xyz-trial_coordinates[atom].xyz)/step.values.x;velocities[atom]=vec4<f32>(v,velocities[atom].w);return;}
 var converged=false;
 for(var iteration=0u;iteration<200u;iteration++){
  var max_error=0.0;
  for(var local=0u;local<group.count;local++){
   let c=constraints[group.start+local];let a=c.atoms.x;let b=c.atoms.y;
   let delta=coordinates[a].xyz-coordinates[b].xyz;let distance_now=length(delta);
   let old_delta=trial_coordinates[step.size.x+a].xyz-trial_coordinates[step.size.x+b].xyz;
   let target_length=c.parameters.x;
   let relative_error=abs(distance_now-target_length)/max(target_length,1e-8);
   max_error=max(max_error,relative_error);
   let direction_projection=dot(delta,old_delta);
   let denominator=(c.parameters.y+c.parameters.z)*direction_projection;
   if(abs(denominator)>1e-20){
    let constraint_delta=0.5*(target_length*target_length-dot(delta,delta))/denominator;
    let correction=constraint_delta*old_delta;
    coordinates[a]=vec4<f32>(coordinates[a].xyz+c.parameters.y*correction,1.0);
    coordinates[b]=vec4<f32>(coordinates[b].xyz-c.parameters.z*correction,1.0);
   }else{atomicOr(&invalid,1u);}
  }
  if(max_error<=1e-5){converged=true;break;}
 }
 if(!converged){atomicOr(&invalid,1u);}
 let parent=group.padding.x;let parent_v=velocities[parent].xyz+(coordinates[parent].xyz-trial_coordinates[parent].xyz)/step.values.x;velocities[parent]=vec4<f32>(parent_v,velocities[parent].w);
 for(var local=0u;local<group.count;local++){let atom=constraints[group.start+local].atoms.y;let v=velocities[atom].xyz+(coordinates[atom].xyz-trial_coordinates[atom].xyz)/step.values.x;velocities[atom]=vec4<f32>(v,velocities[atom].w);}
}
@compute @workgroup_size(64)
fn lf_velocity_correction(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=step.size.x){return;}
 let v=velocities[i].xyz+(coordinates[i].xyz-trial_coordinates[i].xyz)/step.values.x;
 velocities[i]=vec4<f32>(v,velocities[i].w);
}
