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
  var k_eps = max(k, 0.0001);
  return vec4f(col1, d1);
}

fn op_smooth_subtraction(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var k_eps = max(k, 0.0001);
  return vec4f(col1, d1);
}

fn op_smooth_intersection(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var k_eps = max(k, 0.0001);
  return vec4f(col1, d1);
}

fn op(op: f32, d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
// Performs a combination operation on two distances and their corresponding colors.
// 
// # Parameters
// - `op`: The operation type (e.g., union, intersection, difference).
// - `d1`: The first distance value.
// - `d2`: The second distance value.
// - `col1`: The color associated with the first distance.
// - `col2`: The color associated with the second distance.
// - `k`: A blending factor or smoothing parameter.
//
// # Returns
// A `vec4f` containing the resulting distance and blended color.
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
// This function represents the scene in a raymarching shader.
// It takes a 3D position vector `p` as input and returns a 4D vector.
// The returned vector contains the RGB color values in the xyz components
// and the distance to the nearest surface in the w component.
    var d = mix(100.0, p.y, uniforms[17]);

    var spheresCount = i32(uniforms[2]);
    var boxesCount = i32(uniforms[3]);
    var torusCount = i32(uniforms[4]);

    var all_objects_count = spheresCount + boxesCount + torusCount;
    var result = vec4f(vec3f(1.0), d);

    for (var i = 0; i < all_objects_count; i = i + 1)
    {
      // get shape and shape order (shapesinfo)
      // shapesinfo has the following format:
      // x: shape type (0: sphere, 1: box, 2: torus)
      // y: shape index
      // order matters for the operations, they're sorted on the CPU side

      // call transform_p and the sdf for the shape
      // call op function with the shape operation
      var shape = shapesb[i];
      var shape_info = shapesinfob[i];

      // op format:
      // x: operation (0: union, 1: subtraction, 2: intersection)
      // y: k value
      // z: repeat mode (0: normal, 1: repeat)
      // w: repeat offset
      var p_t = transform_p(p, vec2f(shape.op.z, shape.op.w));
      var d : f32;
      
      if (shape_info.x == 0)
      {
        d = sdf_sphere(p_t, shape.radius, shape.quat);
      }
      else if (shape_info.x == 1)
      {
        d = sdf_round_box(p_t, shape.radius.xyz,shape.radius.w, shape.quat);
      }
      else if (shape_info.x == 2)
      {
        d = sdf_torus(p_t, shape.radius.xy, shape.quat);
      }
      
      result = op(shape.op.x, result.w, d, result.xyz, shape.color.xyz, shape.op.y);
      
    }

    return result;
}

fn march(ro: vec3f, rd: vec3f) -> march_output
// Function to perform ray marching
// 
// Parameters:
// - ro: vec3f - The origin of the ray in 3D space
// - rd: vec3f - The direction of the ray in 3D space
// 
// Returns:
// - march_output - The result of the ray marching process
{
  var max_marching_steps = i32(uniforms[5]);
  var EPSILON = uniforms[23];

  var depth = 0.0;
  var color = vec3f(0.0);
  var march_step = uniforms[22];
  var march_offset = uniforms[21];
  var ray_pos = ro;
  
  for (var i = 0; i < max_marching_steps; i = i + 1)
  {

      // raymarch algorithm
      // call scene function and march
      var march_result: vec4f = scene(ray_pos); // xyz = color, w = distance
      depth = march_result.w;
      // if the depth is greater than the max distance or the distance is less than the epsilon, break
      if (depth > MAX_DIST || depth < EPSILON)
      {
        color = vec3f(1.0);
        break;
      }
      ray_pos = ro + rd * depth;
      depth += march_offset;
  }

  return march_output(color, depth, false);
}

fn get_normal(p: vec3f) -> vec3f
{
  var eps = uniforms[24];
  var n = vec3f(0.0);
  n.x = scene(p + vec3f(eps, 0.0, 0.0)).w - scene(p - vec3f(eps, 0.0, 0.0)).w;
  n.y = scene(p + vec3f(0.0, eps, 0.0)).w - scene(p - vec3f(0.0, eps, 0.0)).w;
  n.z = scene(p + vec3f(0.0, 0.0, eps)).w - scene(p - vec3f(0.0, 0.0, eps)).w;
  return normalize(n);
}

// https://iquilezles.org/articles/rmshadows/
fn get_soft_shadow(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, k: f32) -> f32
{
  return 0.0;
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

  // calculate light based on the normal
  // if the object is too far away from the light source, return ambient light
  if (length(current) > uniforms[20] + uniforms[8])
  {
    return ambient;
  }

  var light_direction = normalize(light_position - current);
  var col = saturate(dot(normal, light_direction));

  // calculate the light intensity
  // Use:
  // - shadow
  // - ambient occlusion (optional)
  // - ambient light
  // - object color
  return ambient + col;
}

fn set_camera(ro: vec3f, ta: vec3f, cr: f32) -> mat3x3<f32>
{
  var cw = normalize(ta - ro);
  var cp = vec3f(sin(cr), cos(cr), 0.0);
  var cu = normalize(cross(cw, cp));
  var cv = normalize(cross(cu, cw));
  return mat3x3<f32>(cu, cv, cw);
}

fn animate(val: vec3f, time_scale: f32, offset: f32) -> vec3f
{
  return vec3f(0.0);
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
  var march_result: march_output = march(ro, rd);
  var color = march_result.color;
  
  // display the result
  color = linear_to_gamma(color);
  fb[mapfb(id.xy, uniforms[1])] = vec4(color, 1.0);
}