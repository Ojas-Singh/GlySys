struct Config { sizes:vec4<u32>, values:vec4<f32> }
struct Pose { bounds:vec4<u32>, indices:vec4<u32>, b:vec4<f32>, link:vec4<f32> }
struct Update { value:vec4<f32>, indices:vec4<u32> }
struct StericBounds { minimum:vec3<f32>, maximum:vec3<f32> }
@group(0) @binding(0) var<uniform> config:Config;
@group(0) @binding(1) var<storage,read> protein:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> poses:array<Pose>;
@group(0) @binding(3) var<storage,read> library:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> genes:array<vec4<u32>>;
@group(0) @binding(5) var<storage,read_write> coordinates:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> scores:array<f32>;
@group(0) @binding(7) var<storage,read> updates:array<Update>;
@group(0) @binding(8) var<storage,read> grid:array<vec4<u32>>;
fn rotate(p:vec3<f32>,origin:vec3<f32>,axis:vec3<f32>,angle:f32)->vec3<f32>{let u=normalize(axis);let v=p-origin;return origin+v*cos(angle)+cross(u,v)*sin(angle)+u*dot(u,v)*(1.0-cos(angle));}
fn dihedral(a:vec3<f32>,b:vec3<f32>,c:vec3<f32>,d:vec3<f32>)->f32{let u=normalize(c-b);let v=-(b-a)-dot(-(b-a),u)*u;let w=(d-c)-dot(d-c,u)*u;return atan2(dot(cross(u,v),w),dot(v,w));}
@compute @workgroup_size(64)
fn transform(@builtin(global_invocation_id) id:vec3<u32>){
 let index=id.x;if(index>=config.sizes.x*config.sizes.y){return;}
 let gene=genes[index];let pose=poses[gene.x];let candidate=index/config.sizes.x;
 let base=candidate*config.sizes.z+gene.w;let psi=bitcast<f32>(gene.z)-pose.link.w;
 let c1=rotate(library[pose.bounds.x+pose.bounds.w].xyz,pose.link.xyz,pose.link.xyz-pose.b.xyz,psi);
 let o5=rotate(library[pose.bounds.x+pose.indices.x].xyz,pose.link.xyz,pose.link.xyz-pose.b.xyz,psi);
 let phi=bitcast<f32>(gene.y)-dihedral(pose.b.xyz,pose.link.xyz,c1,o5);
 for(var i=0u;i<pose.bounds.y;i++){
  let p=rotate(library[pose.bounds.x+i].xyz,pose.link.xyz,pose.link.xyz-pose.b.xyz,psi);
  coordinates[base+i]=vec4<f32>(rotate(p,pose.link.xyz,c1-pose.link.xyz,phi),0.0);
 }
}
fn protein_point(candidate:u32,index:u32)->vec3<f32>{
 var p=protein[index].xyz;
 for(var site=0u;site<config.sizes.x;site++) {let pose=poses[genes[candidate*config.sizes.x+site].x];for(var u=pose.indices.y;u<pose.indices.z;u++){if(updates[u].indices.x==index){p=updates[u].value.xyz;}}}
 return p;
}
fn key_less(a:vec3<i32>,b:vec3<i32>)->bool {
 return a.x<b.x || (a.x==b.x && (a.y<b.y || (a.y==b.y && a.z<b.z)));
}
fn cell_range(key:vec3<i32>)->vec2<u32> {
 var lower=0u;var upper=grid[0].x;
 loop {if(lower>=upper){break;}let middle=(lower+upper)/2u;let current=bitcast<vec3<i32>>(grid[1u+2u*middle].xyz);if(key_less(current,key)){lower=middle+1u;}else{upper=middle;}}
 if(lower<grid[0].x && all(bitcast<vec3<i32>>(grid[1u+2u*lower].xyz)==key)) {let start=grid[1u+2u*lower].w;return vec2<u32>(start,start+grid[2u+2u*lower].x);}
 return vec2<u32>(0u);
}
fn indexed_protein_score(p:vec3<f32>,candidate:u32,initial:f32)->f32 {
 var score=initial;
 let key=vec3<i32>(floor(p/3.4));
 for(var x=-1;x<=1;x++){for(var y=-1;y<=1;y++){for(var z=-1;z<=1;z++){
  let range=cell_range(key+vec3<i32>(x,y,z));
  for(var cursor=range.x;cursor<range.y;cursor++){
   let atom=grid[cursor].x;let delta=p-protein[atom].xyz;let d2=dot(delta,delta);
   if(abs(d2-config.values.x)<0.002){return -1.;}
   if(d2<config.values.x){score+=200.*exp(-d2);if(score>2.){return score;}}
  }
 }}}
 for(var cursor=grid[0].y;cursor<grid[0].y+grid[0].z;cursor++){
  let atom=grid[cursor].x;let delta=p-protein_point(candidate,atom);let d2=dot(delta,delta);
  if(abs(d2-config.values.x)<0.002){return -1.;}
  if(d2<config.values.x){score+=200.*exp(-d2);if(score>2.){return score;}}
 }
 return score;
}
fn site_bounds(candidate:u32,site:u32)->StericBounds {
 let gene=genes[candidate*config.sizes.x+site];let pose=poses[gene.x];
 let base=candidate*config.sizes.z+gene.w;
 var minimum=vec3<f32>(1000000.);var maximum=vec3<f32>(-1000000.);
 for(var i=pose.bounds.z;i<pose.bounds.y;i++){
  let p=coordinates[base+i].xyz;minimum=min(minimum,p);maximum=max(maximum,p);
 }
 return StericBounds(minimum,maximum);
}
@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) id:vec3<u32>){
 let index=id.x;if(index>=config.sizes.x*config.sizes.y){return;}
 let candidate=index/config.sizes.x;let site=index%config.sizes.x;let gene=genes[index];let pose=poses[gene.x];let base=candidate*config.sizes.z+gene.w;var score=1.0;
 for(var i=pose.bounds.z;i<pose.bounds.y;i++){
  let p=coordinates[base+i].xyz;
  if(grid[0].w==1u && config.values.x<=2.89 && all(abs(p)<vec3<f32>(100000.))) {
   score=indexed_protein_score(p,candidate,score);if(score<0.){scores[index]=-1.;return;}
  } else {
  for(var j=0u;j<config.sizes.w;j++){let delta=p-protein_point(candidate,j);let d2=dot(delta,delta);if(abs(d2-config.values.x)<0.002){scores[index]=-1.0;return;}if(d2<config.values.x){score+=200.0*exp(-d2);if(score>2.0){break;}}}
 }
 if(score>2.0){break;}
 }
 let own_bounds=site_bounds(candidate,site);let cutoff=sqrt(config.values.x);
 for(var other=0u;other<config.sizes.x;other++){
  if(other==site||score>2.0){continue;}
  let g=genes[candidate*config.sizes.x+other];let op=poses[g.x];var pair=1.0;
  let other_bounds=site_bounds(candidate,other);
  if(any(own_bounds.maximum+vec3<f32>(cutoff)<=other_bounds.minimum)||any(other_bounds.maximum+vec3<f32>(cutoff)<=own_bounds.minimum)){continue;}
  for(var i=pose.bounds.z;i<pose.bounds.y;i++){
   for(var j=0u;j<op.bounds.y;j++){let delta=coordinates[base+i].xyz-coordinates[candidate*config.sizes.z+g.w+j].xyz;let d2=dot(delta,delta);if(abs(d2-config.values.x)<0.002){scores[index]=-1.0;return;}if(d2<config.values.x){pair+=200.0*exp(-d2);if(pair>2.0){break;}}}
   if(pair>2.0){break;}
  }
  score=max(score,pair);
 }
 scores[index]=score;
}
