struct Atom { position:vec4<f32>, ff:vec4<f32> }
struct Pose { oxygen:vec4<f32>, h1:vec4<f32>, h2:vec4<f32> }
struct Config { atoms:u32, poses:u32, cutoff:f32, spare:u32 }
@group(0) @binding(0) var<uniform> config:Config;
@group(0) @binding(1) var<storage,read> receptor:array<Atom>;
@group(0) @binding(2) var<storage,read> poses:array<Pose>;
@group(0) @binding(3) var<storage,read_write> scores:array<vec4<f32>>;
var<workgroup> partial:array<vec4<f32>,64>;
@compute @workgroup_size(64)
fn evaluate(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let pose=poses[group.x];
    var total=vec4<f32>(0.);
    for(var a=lane;a<config.atoms;a+=64u) {
        let atom=receptor[a];
        for(var i=0u;i<3u;i++) {
            var p=pose.oxygen.xyz; var charge=-0.834;
            if(i==1u){p=pose.h1.xyz;charge=0.417;}
            if(i==2u){p=pose.h2.xyz;charge=0.417;}
            let delta=p-atom.position.xyz;
            let d2=dot(delta,delta);
            if(d2<0.25){total.z=1.;}
            if(config.cutoff>0. && d2>config.cutoff*config.cutoff){continue;}
            let inverse=inverseSqrt(max(d2,0.25));
            if(i==0u){
                let r=(atom.ff.y+1.7683)*inverse;
                let r2=r*r;let r6=r2*r2*r2;
                total.x+=sqrt(atom.ff.z*0.1520)*(r6*r6-2.*r6);
            }
            total.y+=332.063713299*atom.ff.x*charge*inverse;
        }
    }
    partial[lane]=total;
    workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u){
        if(lane<stride){partial[lane]+=partial[lane+stride];}
        workgroupBarrier();
    }
    if(lane==0u){scores[group.x]=partial[0];}
}
