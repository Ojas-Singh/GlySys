const COMPUTE_GRADIENTS: bool = true;
struct Config { size:vec4<u32>, energy:vec4<f32>, solvent:vec4<f32>, spare:vec4<f32> }
struct Atom { ff:vec4<f32>, more:vec4<f32>, ranges:vec4<u32> }
struct Term { ids:vec4<u32>, parameters:vec4<f32>, reference:vec4<f32> }
struct Special { other:u32, scee:f32, scnb:f32, spare:u32 }
struct Output { bonded:vec4<f32>, nonbonded:vec4<f32>, extra:vec4<f32>, gradient:vec4<f32> }
@group(0) @binding(0) var<uniform> config:Config;
@group(0) @binding(1) var<storage,read> atoms:array<Atom>;
@group(0) @binding(2) var<storage,read> coordinates:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> terms:array<Term>;
@group(0) @binding(4) var<storage,read> incidence:array<vec2<u32>>;
@group(0) @binding(5) var<storage,read> specials:array<Special>;
@group(0) @binding(6) var<storage,read_write> born:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> output:array<vec4<f32>>;
const C:f32=332.063713299;
const PI:f32=3.141592653589793;

// Local forward derivatives are three-dimensional, never 3*N dimensional.
struct D { v:f32, g:vec3<f32> }
struct DV { x:D,y:D,z:D }
fn dc(v:f32)->D{return D(v,vec3<f32>(0.0));}
fn da(a:D,b:D)->D{return D(a.v+b.v,a.g+b.g);}
fn ds(a:D,b:D)->D{return D(a.v-b.v,a.g-b.g);}
fn dm(a:D,b:D)->D{return D(a.v*b.v,a.g*b.v+b.g*a.v);}
fn dk(a:D,k:f32)->D{return D(a.v*k,a.g*k);}
fn di(a:D)->D{return D(1.0/a.v,-a.g/(a.v*a.v));}
fn dd(a:D,b:D)->D{return dm(a,di(b));}
fn root(a:D)->D{let v=sqrt(a.v);return D(v,a.g/(2.0*max(v,1e-30)));}
fn floor_d(a:D,m:f32)->D{if(a.v<m){return dc(m);}return a;}
fn cosine(a:D)->D{return D(cos(a.v),-sin(a.v)*a.g);}
// Range reduction avoids device-dependent low-accuracy inverse trig intrinsics.
fn precise_atan2(y:f32,x:f32)->f32 {
 let ax=abs(x);let ay=abs(y);if(max(ax,ay)==0.0){return 0.0;}
 var z=min(ax,ay)/max(ax,ay);var offset=0.0;
 if(z>0.41421356237){z=(z-1.0)/(z+1.0);offset=PI*0.25;}
 let z2=z*z;var term=z;var value=z;
 for(var k=1u;k<12u;k++){term *= -z2;value+=term/f32(2u*k+1u);}
 value+=offset;if(ay>ax){value=PI*0.5-value;}if(x<0.0){value=PI-value;}if(y<0.0){value=-value;}return value;
}
fn atan_d(a:D,b:D)->D{return D(precise_atan2(a.v,b.v),(b.v*a.g-a.v*b.g)/max(a.v*a.v+b.v*b.v,1e-30));}
fn acos_d(a:D)->D{let v=clamp(a.v,-1.0,1.0);let sine=sqrt(max(1.0-v*v,0.0));return D(precise_atan2(sine,v),-a.g/max(sine,1e-12));}
fn sub_v(a:DV,b:DV)->DV{return DV(ds(a.x,b.x),ds(a.y,b.y),ds(a.z,b.z));}
fn dot_v(a:DV,b:DV)->D{return da(da(dm(a.x,b.x),dm(a.y,b.y)),dm(a.z,b.z));}
fn mul_v(a:DV,b:D)->DV{return DV(dm(a.x,b),dm(a.y,b),dm(a.z,b));}
fn cross_v(a:DV,b:DV)->DV{return DV(ds(dm(a.y,b.z),dm(a.z,b.y)),ds(dm(a.z,b.x),dm(a.x,b.z)),ds(dm(a.x,b.y),dm(a.y,b.x)));}
fn point(batch:u32,index:u32,target_atom:u32)->DV{
 let p=coordinates[batch*config.size.x+index].xyz;
 if(COMPUTE_GRADIENTS && index==target_atom){return DV(D(p.x,vec3<f32>(1,0,0)),D(p.y,vec3<f32>(0,1,0)),D(p.z,vec3<f32>(0,0,1)));}
 return DV(dc(p.x),dc(p.y),dc(p.z));
}
fn term_value(t:Term,batch:u32,target_atom:u32)->D{
 let kind=u32(t.parameters.w);
 let a=point(batch,t.ids.x,target_atom);
 if(kind==4u){let r=DV(dc(t.reference.x),dc(t.reference.y),dc(t.reference.z));let d=sub_v(a,r);return dk(dot_v(d,d),t.parameters.x);}
 let b=point(batch,t.ids.y,target_atom);
 if(kind==0u){let d=sub_v(a,b);let delta=ds(root(dot_v(d,d)),dc(t.parameters.y));return dk(dm(delta,delta),t.parameters.x);}
 let c=point(batch,t.ids.z,target_atom);
 if(kind==1u){let left=sub_v(a,b);let right=sub_v(c,b);let angle=acos_d(dd(dot_v(left,right),dm(floor_d(root(dot_v(left,left)),1e-12),floor_d(root(dot_v(right,right)),1e-12))));let delta=ds(angle,dc(t.parameters.y));return dk(dm(delta,delta),t.parameters.x);}
 let d=point(batch,t.ids.w,target_atom);
 let b0=sub_v(a,b);let b1=sub_v(c,b);let b2=sub_v(d,c);
 let unit=mul_v(b1,di(floor_d(root(dot_v(b1,b1)),1e-12)));
 let v=sub_v(b0,mul_v(unit,dot_v(b0,unit)));let w=sub_v(b2,mul_v(unit,dot_v(b2,unit)));
 let phi=atan_d(dot_v(cross_v(unit,v),w),dot_v(v,w));
 return dk(da(dc(1.0),cosine(ds(dk(phi,t.parameters.y),dc(t.parameters.z)))),t.parameters.x);
}
fn arity(kind:u32)->u32{if(kind==0u){return 2u;}if(kind==1u){return 3u;}if(kind==4u){return 1u;}return 4u;}
fn active_term(t:Term,batch:u32)->bool{
 if(config.size.w==0u){return true;}
 for(var k=0u;k<arity(u32(t.parameters.w));k++){if(coordinates[batch*config.size.x+t.ids[k]].w!=0.0){return true;}}
 return false;
}
fn radial(r:f32,s:f32,d:f32)->vec2<f32>{
 if(d+s<=r){return vec2<f32>(0.0);}
 let candidate=abs(d-s);let l=max(r,candidate);let u=d+s;
 if(l>=u){return vec2<f32>(0.0);}
 var dl=select(1.0,-1.0,d<s);if(candidate<r){dl=0.0;}
 let a=1.0/l;let b=1.0/u;let c=d-s*s/d;let q=b*b-a*a;let lg=log(l/u);
 return 0.5*vec2<f32>(a-b+0.25*c*q+0.5*lg/d,-dl*a*a+b*b+0.25*((1.0+s*s/(d*d))*q+c*(-2.0*b*b*b+2.0*dl*a*a*a))+0.5*((dl*a-b)/d-lg/(d*d)));
}
@compute @workgroup_size(64)
fn born_radii(@builtin(global_invocation_id) id:vec3<u32>){
 let n=config.size.x;let index=id.x;if(index>=n*config.size.y){return;}
 let i=index%n;let base=index-i;let r=max(atoms[i].ff.w-0.09,0.1);var sum=0.0;
 for(var tile=0u;tile<n;tile+=64u){for(var j=tile;j<min(tile+64u,n);j++){
  if(i!=j){let d=max(distance(coordinates[index].xyz,coordinates[base+j].xyz),1e-8);sum+=radial(r,max(atoms[j].ff.w-0.09,0.1)*atoms[j].more.x,d).x;}
 }}
 let psi=r*sum;let t=tanh(psi-0.8*psi*psi+4.85*psi*psi*psi);let denominator=1.0/r-t/atoms[i].ff.w;
 let b=1.0/max(denominator,1e-6);var derivative=0.0;
 if(denominator>=1e-6){derivative=b*b*(1.0-t*t)*(1.0-1.6*psi+14.55*psi*psi)*r/atoms[i].ff.w;}
 born[index]=vec4<f32>(b,derivative,0.0,0.0);
}
@compute @workgroup_size(64)
fn born_adjoint(@builtin(global_invocation_id) id:vec3<u32>){
 let n=config.size.x;let index=id.x;if(index>=n*config.size.y){return;}
 let i=index%n;let base=index-i;let bi=born[index].x;var db=0.0;let dielectric=1.0/config.solvent.x-1.0/config.solvent.y;
 for(var j=0u;j<n;j++){
  let r2=dot(coordinates[index].xyz-coordinates[base+j].xyz,coordinates[index].xyz-coordinates[base+j].xyz);
  let bj=born[base+j].x;let p=bi*bj;let e=exp(-r2/(4.0*p));let f=sqrt(r2+p*e);
  if(f<1e-8){continue;}
  let coefficient=-C*dielectric*atoms[i].ff.x*atoms[j].ff.x;
  // The half-weight self term has two equal Born-radius derivatives.
  db+=(-0.5*coefficient/(f*f*f))*e*(1.0+r2/(4.0*p))*bj;
 }
 let radius=atoms[i].ff.w;let ratio=radius/bi;let ratio2=ratio*ratio;
 let surface=4.0*PI*config.solvent.w*(radius+config.solvent.z)*(radius+config.solvent.z)*ratio2*ratio2*ratio2;
 born[index].z=db-6.0*surface/bi;
}
@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) id:vec3<u32>){
 let n=config.size.x;let index=id.x;if(index>=n*config.size.y){return;}
 let i=index%n;let batch=index/n;let base=batch*n;let ai=atoms[i];let ci=coordinates[index];
 var result:Output;result.bonded=vec4<f32>(0.0);result.nonbonded=vec4<f32>(0.0);result.extra=vec4<f32>(0.0);result.gradient=vec4<f32>(0.0);
 var g=vec3<f32>(0.0);
 if(config.size.z==0u){
  for(var term_ref=ai.ranges.x;term_ref<ai.ranges.y;term_ref++){
   let t=terms[incidence[term_ref].x];if(!active_term(t,batch)){continue;}
   let value=term_value(t,batch,i);let kind=u32(t.parameters.w);g+=value.g;
   if(kind==4u){result.extra.x+=value.v;}else{result.bonded[kind]+=value.v/f32(arity(kind));}
  }
 }
 let dielectric=1.0/config.solvent.x-1.0/config.solvent.y;
 for(var tile=0u;tile<n;tile+=64u){for(var j=tile;j<min(tile+64u,n);j++){
  let delta=ci.xyz-coordinates[base+j].xyz;let d2=dot(delta,delta);
  if(config.energy.z!=0.0 && config.size.z==0u){
   let bi=born[index].x;let bj=born[base+j].x;let p=bi*bj;let e=exp(-d2/(4.0*p));let f=max(sqrt(d2+p*e),1e-8);
   let coefficient=-C*dielectric*ai.ff.x*atoms[j].ff.x;
   result.nonbonded.z+=0.5*coefficient/f;
   if(COMPUTE_GRADIENTS && j!=i){
    g+=delta*(-coefficient/(f*f*f))*(1.0-0.25*e);
    let d=sqrt(d2);
    if(d>=1e-8){
     let dri=radial(max(ai.ff.w-0.09,0.1),max(atoms[j].ff.w-0.09,0.1)*atoms[j].more.x,d).y;
     let drj=radial(max(atoms[j].ff.w-0.09,0.1),max(ai.ff.w-0.09,0.1)*ai.more.x,d).y;
     g+=delta*(born[index].z*born[index].y*dri+born[base+j].z*born[base+j].y*drj)/d;
    }
   }
  }
  if(j==i){continue;}
  if(config.size.z!=0u && !(ai.more.y*atoms[j].more.y==2.0)){continue;}
  if(config.size.w!=0u && ci.w==0.0 && coordinates[base+j].w==0.0){continue;}
  if(config.energy.x>0.0 && d2>config.energy.x*config.energy.x){continue;}
  var scee=1.0;var scnb=1.0;
  for(var k=ai.ranges.z;k<ai.ranges.w;k++){
   if(specials[k].other==j){scee=specials[k].scee;scnb=specials[k].scnb;break;}
  }
  if(scee==0.0){continue;}
  let d=max(sqrt(d2),1e-8);let radius=ai.ff.y+atoms[j].ff.y;let epsilon=sqrt(ai.ff.z*atoms[j].ff.z);
  let ratio=radius/d;let ratio2=ratio*ratio;let ratio6=ratio2*ratio2*ratio2;
  let coulomb=C*ai.ff.x*atoms[j].ff.x/(config.energy.y*scee*d);
  result.nonbonded.x+=0.5*epsilon*(ratio6*ratio6-2.0*ratio6)/scnb;
  result.nonbonded.y+=0.5*coulomb;
  result.extra.y+=0.5;
  if(COMPUTE_GRADIENTS){g+=delta*(12.0*epsilon*(ratio6-ratio6*ratio6)/(scnb*d)-coulomb/d)/d;}
 }}
 if(config.energy.z!=0.0 && config.size.z==0u){let ratio=ai.ff.w/born[index].x;let ratio2=ratio*ratio;result.nonbonded.w=4.0*PI*config.solvent.w*(ai.ff.w+config.solvent.z)*(ai.ff.w+config.solvent.z)*ratio2*ratio2*ratio2;}
 if(COMPUTE_GRADIENTS && ci.w!=0.0){result.gradient=vec4<f32>(g,0.0);}
 let total=n*config.size.y;
 output[index]=result.bonded;output[total+index]=result.nonbonded;output[2u*total+index]=result.extra;output[3u*total+index]=result.gradient;
}

// Dynamics variant: eight lanes cooperate on each target atom. The original
// score kernels above remain unchanged; this variant increases occupancy for
// medium-sized all-pairs systems such as solvated biomolecules in implicit
// solvent while preserving the same pair and bonded expressions.
var<workgroup> md_scalar:array<f32,64>;
var<workgroup> md_bonded:array<vec4<f32>,64>;
var<workgroup> md_nonbonded:array<vec4<f32>,64>;
var<workgroup> md_extra:array<vec4<f32>,64>;
var<workgroup> md_gradient:array<vec4<f32>,64>;
const MD_LANES_PER_TARGET:u32=8u;
const MD_WORKGROUP_SIZE:u32=64u;
fn md_pair_scales(ranges:vec4<u32>,other:u32)->vec2<f32>{
 var low=ranges.z;var high=ranges.w;
 for(var iteration=0u;iteration<32u && low<high;iteration++){let middle=low+(high-low)/2u;if(specials[middle].other<other){low=middle+1u;}else{high=middle;}}
 if(low<ranges.w && specials[low].other==other){return vec2<f32>(specials[low].scee,specials[low].scnb);}return vec2<f32>(1.0,1.0);
}

fn md_reduce_scalar(local:u32){
 let base=(local/MD_LANES_PER_TARGET)*MD_LANES_PER_TARGET;let lane=local%MD_LANES_PER_TARGET;
 var stride=MD_LANES_PER_TARGET/2u;loop{if(lane<stride){md_scalar[base+lane]+=md_scalar[base+lane+stride];}workgroupBarrier();if(stride==1u){break;}stride/=2u;}
}
fn md_reduce_outputs(local:u32){
 let base=(local/MD_LANES_PER_TARGET)*MD_LANES_PER_TARGET;let lane=local%MD_LANES_PER_TARGET;
 var stride=MD_LANES_PER_TARGET/2u;loop{if(lane<stride){md_bonded[base+lane]+=md_bonded[base+lane+stride];md_nonbonded[base+lane]+=md_nonbonded[base+lane+stride];md_extra[base+lane]+=md_extra[base+lane+stride];md_gradient[base+lane]+=md_gradient[base+lane+stride];}workgroupBarrier();if(stride==1u){break;}stride/=2u;}
}

@compute @workgroup_size(MD_WORKGROUP_SIZE)
fn born_radii_md(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32){
 let n=config.size.x;let total=n*config.size.y;let idx=group.x*(MD_WORKGROUP_SIZE/MD_LANES_PER_TARGET)+local/MD_LANES_PER_TARGET;let lane=local%MD_LANES_PER_TARGET;var sum=0.0;
 if(idx<total){let i=idx%n;let base=idx-i;let r=max(atoms[i].ff.w-0.09,0.1);
  for(var j=lane;j<n;j+=MD_LANES_PER_TARGET){if(i!=j){let d=max(distance(coordinates[idx].xyz,coordinates[base+j].xyz),1e-8);sum+=radial(r,max(atoms[j].ff.w-0.09,0.1)*atoms[j].more.x,d).x;}}
 }
 md_scalar[local]=sum;workgroupBarrier();md_reduce_scalar(local);
 if(lane==0u && idx<total){let i=idx%n;let r=max(atoms[i].ff.w-0.09,0.1);let psi=r*md_scalar[local];let t=tanh(psi-0.8*psi*psi+4.85*psi*psi*psi);let denominator=1.0/r-t/atoms[i].ff.w;let b=1.0/max(denominator,1e-6);var derivative=0.0;if(denominator>=1e-6){derivative=b*b*(1.0-t*t)*(1.0-1.6*psi+14.55*psi*psi)*r/atoms[i].ff.w;}born[idx]=vec4<f32>(b,derivative,0.0,0.0);}
}

@compute @workgroup_size(MD_WORKGROUP_SIZE)
fn born_adjoint_md(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32){
 let n=config.size.x;let total=n*config.size.y;let idx=group.x*(MD_WORKGROUP_SIZE/MD_LANES_PER_TARGET)+local/MD_LANES_PER_TARGET;let lane=local%MD_LANES_PER_TARGET;var db=0.0;
 if(idx<total){let i=idx%n;let base=idx-i;let bi=born[idx].x;let dielectric=1.0/config.solvent.x-1.0/config.solvent.y;
  for(var j=lane;j<n;j+=MD_LANES_PER_TARGET){let delta=coordinates[idx].xyz-coordinates[base+j].xyz;let r2=dot(delta,delta);let bj=born[base+j].x;let p=bi*bj;let e=exp(-r2/(4.0*p));let f=sqrt(r2+p*e);if(f>=1e-8){let coefficient=-C*dielectric*atoms[i].ff.x*atoms[j].ff.x;db+=(-0.5*coefficient/(f*f*f))*e*(1.0+r2/(4.0*p))*bj;}}
 }
 md_scalar[local]=db;workgroupBarrier();md_reduce_scalar(local);
 if(lane==0u && idx<total){let i=idx%n;let bi=born[idx].x;let radius=atoms[i].ff.w;let ratio=radius/bi;let ratio2=ratio*ratio;let surface=4.0*PI*config.solvent.w*(radius+config.solvent.z)*(radius+config.solvent.z)*ratio2*ratio2*ratio2;born[idx].z=md_scalar[local]-6.0*surface/bi;}
}

@compute @workgroup_size(MD_WORKGROUP_SIZE)
fn evaluate_md(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32){
 let n=config.size.x;let total=n*config.size.y;let idx=group.x*(MD_WORKGROUP_SIZE/MD_LANES_PER_TARGET)+local/MD_LANES_PER_TARGET;let lane=local%MD_LANES_PER_TARGET;let valid=idx<total;let i=idx%n;let batch=idx/n;let base=batch*n;
 var bonded_energy=vec4<f32>(0.0);var nonbonded_energy=vec4<f32>(0.0);var extra_energy=vec4<f32>(0.0);var g=vec3<f32>(0.0);
 if(valid && lane==0u && config.size.z==0u){let ai=atoms[i];for(var term_ref=ai.ranges.x;term_ref<ai.ranges.y;term_ref++){let t=terms[incidence[term_ref].x];if(!active_term(t,batch)){continue;}let value=term_value(t,batch,i);let kind=u32(t.parameters.w);g+=value.g;if(kind==4u){extra_energy.x+=value.v;}else{bonded_energy[kind]+=value.v/f32(arity(kind));}}}
 if(valid){let ai=atoms[i];let ci=coordinates[idx];let dielectric=1.0/config.solvent.x-1.0/config.solvent.y;
  for(var j=lane;j<n;j+=MD_LANES_PER_TARGET){let delta=ci.xyz-coordinates[base+j].xyz;let d2=dot(delta,delta);
   if(config.energy.z!=0.0 && config.size.z==0u){let bi=born[idx].x;let bj=born[base+j].x;let p=bi*bj;let e=exp(-d2/(4.0*p));let f=max(sqrt(d2+p*e),1e-8);let coefficient=-C*dielectric*ai.ff.x*atoms[j].ff.x;nonbonded_energy.z+=0.5*coefficient/f;if(j!=i){g+=delta*(-coefficient/(f*f*f))*(1.0-0.25*e);let d=sqrt(d2);if(d>=1e-8){let dri=radial(max(ai.ff.w-0.09,0.1),max(atoms[j].ff.w-0.09,0.1)*atoms[j].more.x,d).y;let drj=radial(max(atoms[j].ff.w-0.09,0.1),max(ai.ff.w-0.09,0.1)*ai.more.x,d).y;g+=delta*(born[idx].z*born[idx].y*dri+born[base+j].z*born[base+j].y*drj)/d;}}}
   if(j==i){continue;}if(config.size.z!=0u && !(ai.more.y*atoms[j].more.y==2.0)){continue;}if(config.size.w!=0u && ci.w==0.0 && coordinates[base+j].w==0.0){continue;}if(config.energy.x>0.0 && d2>config.energy.x*config.energy.x){continue;}
   var scales=md_pair_scales(ai.ranges,j);if(config.spare.y!=0.0){let lookup_base=bitcast<u32>(config.spare.x);let pair=specials[lookup_base+i*n+j];scales=vec2<f32>(pair.scee,pair.scnb);}let scee=scales.x;let scnb=scales.y;if(scee==0.0){continue;}
   let d=max(sqrt(d2),1e-8);let radius=ai.ff.y+atoms[j].ff.y;let epsilon=sqrt(ai.ff.z*atoms[j].ff.z);let ratio=radius/d;let ratio2=ratio*ratio;let ratio6=ratio2*ratio2*ratio2;let coulomb=C*ai.ff.x*atoms[j].ff.x/(config.energy.y*scee*d);nonbonded_energy.x+=0.5*epsilon*(ratio6*ratio6-2.0*ratio6)/scnb;nonbonded_energy.y+=0.5*coulomb;extra_energy.y+=0.5;if(COMPUTE_GRADIENTS){g+=delta*(12.0*epsilon*(ratio6-ratio6*ratio6)/(scnb*d)-coulomb/d)/d;}
  }
 }
 if(valid && lane==0u && config.energy.z!=0.0 && config.size.z==0u){let ai=atoms[i];let ratio=ai.ff.w/born[idx].x;let ratio2=ratio*ratio;nonbonded_energy.w=4.0*PI*config.solvent.w*(ai.ff.w+config.solvent.z)*(ai.ff.w+config.solvent.z)*ratio2*ratio2*ratio2;}
 md_bonded[local]=bonded_energy;md_nonbonded[local]=nonbonded_energy;md_extra[local]=extra_energy;md_gradient[local]=vec4<f32>(g,0.0);workgroupBarrier();md_reduce_outputs(local);
 if(lane==0u && valid){let out_bonded=md_bonded[local];let out_nonbonded=md_nonbonded[local];let out_extra=md_extra[local];let out_gradient=md_gradient[local];output[idx]=out_bonded;output[total+idx]=out_nonbonded;output[2u*total+idx]=out_extra;output[3u*total+idx]=out_gradient;}
}

var<workgroup> partial_a:array<vec4<f32>,64>;
var<workgroup> partial_b:array<vec4<f32>,64>;
var<workgroup> partial_c:array<vec4<f32>,64>;
@compute @workgroup_size(64)
fn reduce(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let total=config.size.x*config.size.y;let base=group.x*config.size.x;
 var a=vec4<f32>(0.0);var b=vec4<f32>(0.0);var c=vec4<f32>(0.0);
 var ca=vec4<f32>(0.0);var cb=vec4<f32>(0.0);var cc=vec4<f32>(0.0);
 for(var i=lane;i<config.size.x;i+=64u){
  let va=output[base+i]-ca;let ta=a+va;ca=(ta-a)-va;a=ta;
  let vb=output[total+base+i]-cb;let tb=b+vb;cb=(tb-b)-vb;b=tb;
  let vc=output[2u*total+base+i]-cc;let tc=c+vc;cc=(tc-c)-vc;c=tc;
 }
 partial_a[lane]=a;partial_b[lane]=b;partial_c[lane]=c;
 workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){
  if(lane<stride){partial_a[lane]+=partial_a[lane+stride];partial_b[lane]+=partial_b[lane+stride];partial_c[lane]+=partial_c[lane+stride];}
  workgroupBarrier();
 }
 if(lane==0u){output[4u*total+3u*group.x]=partial_a[0];output[4u*total+3u*group.x+1u]=partial_b[0];output[4u*total+3u*group.x+2u]=partial_c[0];}
}
