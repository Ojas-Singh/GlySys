struct Step { size:vec4<u32>, values:vec4<f32> }
@group(0) @binding(0) var<uniform> step:Step;
@group(0) @binding(1) var<storage,read_write> coordinates:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> velocities:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> forces:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> noise:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> invalid:atomic<u32>;
@compute @workgroup_size(64)
fn before_force(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=step.size.x){return;}
 let dt=step.values.x;let decay=step.values.y;let invmass=velocities[i].w;
 var v=velocities[i].xyz-0.5*dt*418.4*invmass*forces[3u*step.size.x+i].xyz;
 var movement=0.5*dt*v;
 v=decay*v+sqrt(step.values.z*invmass)*noise[step.size.y+i].xyz;
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
