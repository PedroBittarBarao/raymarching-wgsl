const THREAD_COUNT = 16;
const PI = 3.1415927f;
const MAX_DIST = 1000.0;

@group(0) @binding(0)  
  var<storage, read_write> fb : array<vec4f>;

@group(1) @binding(0)
  var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)
  var<storage, read_write> shapesb : array<shape>;

@group(2) @binding(1)
  var<storage, read_write> shapesinfob : array<vec4f>;

struct shape {
  transform : vec4f, // xyz = position
  radius : vec4f, // xyz = scale, w = global scale
  rotation : vec4f, // xyz = rotation
  op : vec4f, // x = operation, y = k value, z = repeat mode, w = repeat offset
  color : vec4f, // xyz = color
  animate_transform : vec4f, // xyz = animate position value (sin amplitude), w = animate speed
  animate_rotation : vec4f, // xyz = animate rotation value (sin amplitude), w = animate speed
  quat : vec4f, // xyzw = quaternion
  transform_animated : vec4f, // xyz = position buffer
};

struct march_output {
  color : vec3f,
  depth : f32,
  outline : bool,
};

fn op_smooth_union(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
  var d = mix(d2, d1, h) - k * h * (1.0 - h);
  var col = mix(col2, col1, h);
  return vec4f(col, d);
}

fn op_smooth_subtraction(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var k_eps = max(k, 0.0001);
  var h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
  var d = mix(d2, -d1, h) + k * h * (1.0 - h);
  var col = mix(col2, col1, h);
  return vec4f(col, d);
}

fn op_smooth_intersection(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
  var d = mix(d2, d1, h) + k * h * (1.0 - h);
  var col = mix(col2, col1, h);
  return vec4f(col, d);
}

fn op(op: f32, d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f

{
  // union
  if (op < 1.0)
  {
    return op_smooth_union(d1, d2, col1, col2, k);
  }

  // subtraction
  if (op < 2.0)
  {
    return op_smooth_subtraction(d2, d1, col2, col1, k);
  }

  // intersection
  return op_smooth_intersection(d2, d1, col2, col1, k);
}

fn repeat(p: vec3f, offset: vec3f) -> vec3f
{
  return modc(p,offset);
}

fn transform_p(p: vec3f, option: vec2f) -> vec3f
{
  // normal mode
  if (option.x <= 1.0)
  {
    return p;
  }

  // return repeat / mod mode
  return repeat(p, vec3f(option.y));
}

fn scene(p: vec3f) -> vec4f // xyz = color, w = distance
{
    var d = mix(100.0, p.y, uniforms[17]);

    var spheresCount = i32(uniforms[2]);
    var boxesCount = i32(uniforms[3]);
    var torusCount = i32(uniforms[4]);

    var all_objects_count = spheresCount + boxesCount + torusCount;
    var result = vec4f(vec3f(1.0), d);

    for (var i = 0; i < all_objects_count; i = i + 1)
    {
      // get shape and shape order (shapesinfo)
      var shape_info = shapesinfob[i];
      let shape_index = i32(shape_info.y);
      let shape_type = shape_info.x; 
      var shape = shapesb[shape_index];
      let animate_transform = shape.animate_transform;
      let animate_rotation = shape.animate_rotation;

      // shapesinfo has the following format:
      // x: shape type (0: sphere, 1: box, 2: torus)
      // y: shape index
      let quat = shape.quat;
      let p = p - shape.transform_animated.xyz;

      if ( shape_type > 1.0) // torus
      { 
        d = sdf_torus(p,shape.radius.xy,quat); 
      }
      else if (shape_type > 0.0)// box
      {
        d = sdf_round_box(p, shape.radius.xyz, shape.radius.w, quat);
      } 
      else  // sphere
      {
        d = sdf_sphere(p,shape.radius,quat);
      }
      
      if (d < result.w) // if closest object 
      {
        result.w = d; // assign closest distance
        let res = vec4f(shape.color.xyz,d);
        result = res; // assign color and distance 
      }


      let op_type = shape.op.x;
      let k = shape.op.y;
      let d1 = d;
      let d2 = result.w;

      let c1 = shape.color.xyz;
      let c2 = result.xyz;
      
      let op_res = op(op_type,d1,d2,c1,c2,k);

      let op_col = op_res.xyz;
      let op_d = op_res.w;

      if (op_d < result.w) // return smallest distance
      {
        let res = vec4f(op_col,op_d);
        result = res; // assign color and distance 
      }

    }

    return result;
}

fn march(ro: vec3f, rd: vec3f) -> march_output
{
  var max_marching_steps = i32(uniforms[5]);
  var EPSILON = uniforms[23];

  var depth = 0.0;
  var color = vec3f(0.0);
  var distance = 10000;
  var march_step = uniforms[22];
  
  for (var i = 0; i < max_marching_steps * 3; i = i + 1) // max_marching_steps * 3 to fix horizon issue
  {
      // raymarch algorithm
      let p = ro+rd*depth;
      // call scene function and march
      let res = scene(p);
      let distance = res.w;
      color = res.xyz;
      // how to determine color
      // if the depth is greater than the max distance or the distance is less than the epsilon, break
      if (depth > MAX_DIST || distance < EPSILON)
      {
        return march_output(color,depth,false);
      }
      depth += distance;
  }
  return march_output(vec3f(0.0), MAX_DIST, false);
  
}

fn get_normal(p: vec3f) -> vec3f
{
    var eps = 0.0001;
    var h = vec2f(eps, 0.0);
    return normalize(vec3f(
        scene(p + h.xyy).w - scene(p - h.xyy).w,
        scene(p + h.yxy).w - scene(p - h.yxy).w,
        scene(p + h.yyx).w - scene(p - h.yyx).w
    ));
}

// https://iquilezles.org/articles/rmshadows/
fn get_soft_shadow(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, k: f32) -> f32
{
  var res = 1.0;
  var t = tmin;
  let max_steps = 100;
  var eps = 0.0001;
  for (var i = 0; i < max_steps && t < tmax; i = i + 1)
  {
    let p = ro + rd * t;
    let h = scene(p).w;
    if (h < eps){
      return 0.0;
    }
    res = min(res, k * h / t);
    t += h; 
  }

  return clamp(res, 0.0, 1.0);
}

fn get_AO(current: vec3f, normal: vec3f) -> f32
{
  var occ = 0.0;
  var sca = 1.0;
  for (var i = 0; i < 5; i = i + 1)
  {
    var h = 0.001 + 0.15 * f32(i) / 4.0;
    var d = scene(current + h * normal).w;
    occ += (h - d) * sca;
    sca *= 0.95;
  }

  return clamp( 1.0 - 2.0 * occ, 0.0, 1.0 ) * (0.5 + 0.5 * normal.y);
}

fn get_ambient_light(light_pos: vec3f, sun_color: vec3f, rd: vec3f) -> vec3f
{
  var backgroundcolor1 = int_to_rgb(i32(uniforms[12]));
  var backgroundcolor2 = int_to_rgb(i32(uniforms[29]));
  var backgroundcolor3 = int_to_rgb(i32(uniforms[30]));
  
  var ambient = backgroundcolor1 - rd.y * rd.y * 0.5;
  ambient = mix(ambient, 0.85 * backgroundcolor2, pow(1.0 - max(rd.y, 0.0), 4.0));

  var sundot = clamp(dot(rd, normalize(vec3f(light_pos))), 0.0, 1.0);
  var sun = 0.25 * sun_color * pow(sundot, 5.0) + 0.25 * vec3f(1.0,0.8,0.6) * pow(sundot, 64.0) + 0.2 * vec3f(1.0,0.8,0.6) * pow(sundot, 512.0);
  ambient += sun;
  ambient = mix(ambient, 0.68 * backgroundcolor3, pow(1.0 - max(rd.y, 0.0), 16.0));

  return ambient;
}
fn get_light(current: vec3f, obj_color: vec3f, rd: vec3f) -> vec3f
// Computes the lighting at a given point in the scene.
// 
// Parameters:
// - current: The current position in the scene as a 3D vector.
// - obj_color: The color of the object at the current position as a 3D vector.
// - rd: The ray direction as a 3D vector.
// 
// Returns:
// - A 3D vector representing the computed light color at the given point.
{
  var light_position = vec3f(uniforms[13], uniforms[14], uniforms[15]);
  var sun_color = int_to_rgb(i32(uniforms[16]));
  var ambient = get_ambient_light(light_position, sun_color, rd);
  var normal = get_normal(current);


  let eps = uniforms[24];

  var light_direction = normalize(light_position - current);

  var shadow = get_soft_shadow(current + normal * eps,light_direction,eps,length(light_position - current),8.0);
  var diffuse_intensity = max(dot(normal, light_direction), 0.0);
  var diffuse = diffuse_intensity * obj_color * sun_color;
  diffuse *= shadow;
  var ao = get_AO(current, normal);

  var color = ambient * obj_color;
  color += diffuse;
  color *= ao;

  // calculate the light intensity
  // Use:
  // - shadow
  // - ambient occlusion (optional)
  // - ambient light
  // - object color
  return color;
}

fn set_camera(ro: vec3f, ta: vec3f, cr: f32) -> mat3x3<f32>
{
  var cw = normalize(ta - ro);
  var cp = vec3f(sin(cr), cos(cr), 0.0);
  var cu = normalize(cross(cw, cp));
  var cv = normalize(cross(cu, cw));
  return mat3x3<f32>(cu, cv, cw);
}


fn animate(val: vec3f, amplitude: vec3f, speed: f32, time: f32) -> vec3f {
    return val + amplitude * sin(speed * time);
}

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn preprocess(@builtin(global_invocation_id) id : vec3u)
{
  var time = uniforms[0];
  var spheresCount = i32(uniforms[2]);
  var boxesCount = i32(uniforms[3]);
  var torusCount = i32(uniforms[4]);
  var all_objects_count = spheresCount + boxesCount + torusCount;

  if (id.x >= u32(all_objects_count))
  {
    return;
  }

  let idx = i32(id.x);
  var shape = shapesb[idx];
  var animated_position = animate(
      shape.transform.xyz,
      shape.animate_transform.xyz,
      shape.animate_transform.w,
      time
  );
  shapesb[idx].transform_animated = vec4f(animated_position, shape.transform.w);

  // Animate rotation
  var animated_rotation = animate(
      shape.rotation.xyz,
      shape.animate_rotation.xyz,
      shape.animate_rotation.w,
      time
  );
  shapesb[idx].quat = quaternion_from_euler(animated_rotation);

  // optional: performance boost
  // Do all the transformations here and store them in the buffer since this is called only once per object and not per pixel
}

@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u)
{
  // unpack data
  var fragCoord = vec2f(f32(id.x), f32(id.y));
  var rez = vec2(uniforms[1]);
  var time = uniforms[0];

  // camera setup
  var lookfrom = vec3(uniforms[6], uniforms[7], uniforms[8]);
  var lookat = vec3(uniforms[9], uniforms[10], uniforms[11]);
  var camera = set_camera(lookfrom, lookat, 0.0);
  var ro = lookfrom;

  // get ray direction
  var uv = (fragCoord - 0.5 * rez) / rez.y;
  uv.y = -uv.y;
  var rd = camera * normalize(vec3(uv, 1.0));

  // call march function and get the color/depth
  // move ray based on the depth
  // get light
  var march_result = march(ro, rd);
  var depth = march_result.depth;
  var color = march_result.color;

  if (depth < MAX_DIST)
  {
    var current = ro + rd * depth;
    color = get_light(current, march_result.color, rd);
  }
  else{
        // Ray missed all objects, use background color
        var light_position = vec3f(uniforms[13], uniforms[14], uniforms[15]);
        var sun_color = int_to_rgb(i32(uniforms[16]));
        color = get_ambient_light(light_position, sun_color, rd);
    }
  
  // display the result
  color = linear_to_gamma(color);
  fb[mapfb(id.xy, uniforms[1])] = vec4(color, 1.0);
}