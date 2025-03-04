use image::ImageReader;
use image::Rgb;
use image::RgbImage;
use image::imageops::crop_imm;
use std::cmp::max;
use std::cmp::min;
use std::env;

const DELTA:i32 = 500;
const DELTA_INNER:i32 = 100;
const MARGIN:i32 = 50;
const SLOPE:i32 = 50;

struct Edge {
    start: i32,
    end: i32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        println!("{}", args.len());
        panic!("args");
    }
    let img_reader = ImageReader::open(args[1].clone());
    let img_rgb:RgbImage = img_reader?.decode()?.into_rgb8();
    // print!("rotate... ");
    let img_rotated:RgbImage = rotate_90_clockwise(&img_rgb);
    // println!("rotated");
    img_rotated.save("rotated-90.png")?;


    // print!("left left edge... ");
    let left_left = y_edge(&img_rotated, 0, DELTA, blue_line);
    // println!("end");
    // print!("right right edge... ");
    let right_right = y_edge(&img_rotated, img_rotated.width().try_into().unwrap(), -DELTA, blue_line);
    // println!("end");

    // println!("ll={}-{} rr={}-{}", left_left.start, left_left.end, right_right.start, right_right.end);

    let middle:i32 = (left_left.start + right_right.start) / 2;
    // print!("left right edge... ");
    let left_right = y_edge(&img_rotated, middle, -DELTA_INNER, blue_line);
    // println!("end");
    // print!("right left edge... ");
    let right_left = y_edge(&img_rotated, middle, DELTA_INNER, blue_line);
    // println!("end");

    // println!("lr={}-{} rl={}-{}", left_right.start, left_right.end, right_left.start, right_left.end);

    let left_image = process(
            &img_rotated,
            &left_left,
            &left_right,
        );
    let right_image = process(
            &img_rotated,
            &right_left,
            &right_right,
        );

    left_image.save(args[2].clone())?;
    right_image.save(args[3].clone())?;

    Ok(())
}

fn angle(x1: i32, y1: i32, x2: i32, y2: i32) -> f32 {
    assert!(y2 != y1);
    let dx: f32 = (x1 - x2) as f32;
    let dy: f32 = (y1 - y2) as f32;
    let angle_tan: f32 = dx / dy;
    angle_tan.atan()
}

fn rotate_90_clockwise(image: &RgbImage) -> RgbImage {
    let mut rotated:RgbImage = RgbImage::new(image.height(), image.width());
    for y in 0 .. image.height() {
        for x in 0 .. image.width() {
            rotated.put_pixel(image.height() - y - 1 , x, *image.get_pixel(x, y));
        }
    }
    rotated
}

fn x_edge(
        image: &RgbImage,
        startx: i32, deltax: i32,
        score: fn(&Vec<Rgb<u8>>) -> i32,
    ) -> Edge {
    generic_edge(image, startx, deltax, score, true)
}

fn y_edge(
        image: &RgbImage,
        startx: i32, deltax: i32,
        score: fn(&Vec<Rgb<u8>>) -> i32,
    ) -> Edge {
    generic_edge(image, startx, deltax, score, false)
}


fn generic_edge(
        image: &RgbImage,
        startx: i32, deltax: i32,
        score: fn(&Vec<Rgb<u8>>) -> i32,
        is_x: bool,
    ) -> Edge {
    let limit = if is_x { image.width() } else { image.height() };
    let mut edge = Vec::<Rgb<u8>>::new();
    let mut deltas = Vec::<i32>::new();
    let mut max_jump = -1;
    let mut max_edge = Edge {start: -1, end: -1};
    for _ in 0 .. limit {
        edge.push(*image.get_pixel(0, 0));
        deltas.push(0);
    }
    let (real_startx, real_endx) = if deltax < 0 {
            (startx + deltax, startx)
        } else {
            (startx, startx + deltax)
        };
    // TODO: Slope arg.
    for slope in -SLOPE .. SLOPE + 1 {
        let current_startx = if slope < 0 { real_startx - slope } else { real_startx };
        let current_endx = if slope > 0 { real_endx - slope } else { real_endx };
        if slope == 0 {
            for i in 0 .. deltas.len() {
                deltas[i] = 0;
            }
        } else {
            // We have a line from (x, 0) to (x+slope, deltas_len)
            // which, for the loop below, is equivalent to a line from
            // (0, 0) to (deltas_len, slope).
            //
            // For a given coordinate y, we can compute x as
            // y * (slope - 0) / (deltas_len - 0) = y * slope / deltas_len.
            //
            // If we know x for a given y, then we can compute x_1 corresponding
            // to y + 1 as y * slope / deltas_len + slope / deltas_len
            //
            // We want to know when the rounded value for x changes, and that
            // happens whenever we add slope/deltas_len and x goes from one
            // bucket to another, where the buckets have length 1, and are of
            // the form [n-0.5, n+0.5). But that's equivalent to adding just
            // "slope" to x and using buckets of length deltas_len.
            let deltas_len_i32: i32 = deltas.len().try_into().unwrap();
            let mut offset: i32 = deltas_len_i32 / 2;
            let mut delta = 0;
            for i in 0 .. deltas.len() {
                offset += slope;
                if offset > deltas_len_i32 {
                    offset -= deltas_len_i32;
                    delta += 1;
                } else if offset < 0 {
                    offset += deltas_len_i32;
                    delta -= 1;
                }
                deltas[i] = delta;
            }
        }
        let mut previous_score: i32 = -1;
        for i in current_startx .. current_endx {
            if is_x {
                let width_usize: usize = image.width().try_into().unwrap();
                for j in 0 .. width_usize {
                    edge[j] = image.get_pixel(j.try_into().unwrap(), (i + deltas[j]).try_into().unwrap()).clone();
                }
            } else {
                for j in 0 .. image.height().try_into().unwrap() {
                    edge[j] = image.get_pixel((i + deltas[j]).try_into().unwrap(), j.try_into().unwrap()).clone();
                }
            }
            let current_score = score(&edge);
            assert!(current_score >= 0);
            if previous_score != -1 && current_score > previous_score {
                let jump = current_score - previous_score;
                if jump > max_jump {
                    max_jump = jump;
                    max_edge = Edge {start: i, end: i + slope};
                }
            }
            previous_score = current_score;
        }
    }
    assert!(max_jump >= 0);
    max_edge
}


fn process(image: &RgbImage, left: &Edge, right: &Edge) -> RgbImage {
  let sstart = min(left.start, left.end);
  let send = max(right.start, right.end);
  let crop_start = max(sstart - MARGIN, 0);
  let crop_end: i32 = min(send + MARGIN, image.width().try_into().unwrap());
  let cropped: RgbImage = crop_imm(
          image,
          crop_start.try_into().unwrap(),
          0,
          (crop_end - crop_start).try_into().unwrap(),
          image.height()
      ).to_image();
  // cropped.save(format!("cropped-{}-{}.png", sstart, send))?;
  let up = x_edge(&cropped, 0, DELTA, blue_line);
  let down = x_edge(&cropped, cropped.height().try_into().unwrap(), -DELTA, blue_line);
  let estart = min(up.start, up.end);
  let eend = max(down.start, down.end);
  let crop_start = max(estart - MARGIN, 0);
  let crop_end: i32 = min(eend + MARGIN, cropped.height().try_into().unwrap());
  let cropped_again = crop_imm(
          &cropped,
          0,
          crop_start.try_into().unwrap(),
          cropped.width(),
          (crop_end - crop_start).try_into().unwrap()
      );
  let angle1 = angle(left.start, 0, left.end, image.height().try_into().unwrap());
  let angle2 = angle(right.start, 0, right.end, image.height().try_into().unwrap());
  let angle3 = angle(-up.start, 0, -up.end, image.height().try_into().unwrap());
  let angle4 = angle(-down.start, 0, -down.end, image.height().try_into().unwrap());
  let angle = (angle1 + angle2 + angle3 + angle4) / 4.0;
  let rotated = rotate(&cropped_again.to_image(), -angle);
  let cleaned = cleanup(&rotated);
  // let scaled = scale_down(&cleaned);
  // cleanup(&scaled)
  cleaned
}

fn cleanup(image: &RgbImage) -> RgbImage {
  // TODO: Send as args.
  let colors: Vec<Rgb<u8>> =
      vec![ Rgb([191, 117, 117]),
            Rgb([102, 102, 191]),
            Rgb([244, 244, 140]),
            Rgb([76, 76, 76]),
            Rgb([244, 244, 244])];
  // TODO: Send as args.
  let replacement_colors: Vec<Rgb<u8>> =
      vec![ Rgb([0, 0, 0]),
            Rgb([0, 0, 0]),
            Rgb([255, 255, 255]),
            Rgb([0, 0, 0]),
            Rgb([255, 255, 255])];
  // let replacement_colors: Vec<Rgb<u8>> =
  //     vec![ Rgb([191, 117, 150]),
  //           Rgb([102, 102, 191]),
  //           Rgb([255, 255, 146]),
  //           Rgb([0, 0, 0]),
  //           Rgb([255, 255, 255])];
  let mut initial: Vec<Vec<usize>> = Vec::new();
  for i in 0 .. image.height() {
    let mut line: Vec<usize> = Vec::new();
    for j in 0 .. image.width() {
      let mut best_idx = colors.len();
      let mut best_score = i32::MAX;
      let pixel = image.get_pixel(j, i);
      for k in 0 .. colors.len() {
        let score = color_distance(&pixel, &colors[k], false /*j == 654 && i == 396*/);
        // println!("{} {}", k, score);
        if score < best_score {
          best_score = score;
          best_idx = k;
        }
      }
      assert!(best_idx < colors.len());
      line.push(best_idx);
    }
    initial.push(line);
  }
  let mut cleaned = RgbImage::new(image.width(), image.height());
  // TODO: fill the edge.
  let mut color_count: Vec<u8> = Vec::new();
  for _ in 0 .. colors.len() {
    color_count.push(0);
  }
  for i in 1 .. image.width() - 1 {
    // let i_i32: i32 = i.try_into().unwrap();
    let i_usize: usize = i.try_into().unwrap();
    for j in 1 .. image.height() - 1 {
      // let j_i32: i32 = j.try_into().unwrap();
      let j_usize: usize = j.try_into().unwrap();
      for k in 0 .. colors.len() {
        color_count[k] = 0;
      }
      for k in i - 1 .. i + 2 {
        let k_usize: usize = k.try_into().unwrap();
        for l in j - 1 .. j + 2 {
          let l_usize: usize = l.try_into().unwrap();
          color_count[initial[l_usize][k_usize]] += 1;
        }
      }
      let mut max_idx = 0;
      let mut max = color_count[0];
      for k in 0 .. colors.len() {
        if color_count[k] > max {
          max_idx = k;
          max = color_count[k];
        }
      }
      // if max > 5 && max_idx != initial[j_usize][i_usize] {
      //   println!("*");
      // }
      if max > 4 {
        cleaned.put_pixel(i, j, replacement_colors[max_idx]);
      } else {
        cleaned.put_pixel(i, j, replacement_colors[initial[j_usize][i_usize]]);
      }
    }
  }
  cleaned
  // image.clone()
}

fn scale_down(image: &RgbImage) -> RgbImage {
  let mut retv = RgbImage::new(image.width()/2, image.height()/2);
  for x in 0 .. retv.width() {
    for y in 0 .. retv.height() {
      let mut r:i32 = 0;
      let mut g:i32 = 0;
      let mut b:i32 = 0;
      add_pixel(image.get_pixel(2*x    , 2*y    ), &mut r, &mut g, &mut b);
      add_pixel(image.get_pixel(2*x    , 2*y + 1), &mut r, &mut g, &mut b);
      add_pixel(image.get_pixel(2*x + 1, 2*y    ), &mut r, &mut g, &mut b);
      add_pixel(image.get_pixel(2*x + 1, 2*y + 1), &mut r, &mut g, &mut b);
      let p = Rgb([(r / 4).try_into().unwrap(), (g / 4).try_into().unwrap(), (b / 4).try_into().unwrap()]);
      retv.put_pixel(x, y, p);
    }
  }
  retv
}

fn add_pixel(pixel: &Rgb<u8>, r: &mut i32, g: &mut i32, b: &mut i32) {
  let pr: i32 = pixel.0[0].into();
  let pg: i32 = pixel.0[1].into();
  let pb: i32 = pixel.0[2].into();
  *r += pr;
  *g += pg;
  *b += pb;
}

fn color_distance(first: &Rgb<u8>, second: &Rgb<u8>, log: bool) -> i32 {
  let hsv_first = rgb_to_hsv(first);
  let hsv_second = rgb_to_hsv(second);
  let hfirst: i32 = hsv_first.0[0].into();
  let sfirst: i32 = hsv_first.0[1].into();
  let vfirst: i32 = hsv_first.0[2].into();
  if log {
      println!("{}-{}-{} -> {}-{}-{}", first.0[0], first.0[1], first.0[2], hfirst, sfirst, vfirst);
  }
  let hsecond: i32 = hsv_second.0[0].into();
  let ssecond: i32 = hsv_second.0[1].into();
  let vsecond: i32 = hsv_second.0[2].into();
  if log {
      println!("{}-{}-{} -> {}-{}-{}", second.0[0], second.0[1], second.0[2], hsecond, ssecond, vsecond);
  }
  let hdif = min((hfirst - hsecond).abs(), min((hfirst + 360 - hsecond).abs(), (hfirst - (hsecond + 360)).abs()));
  let hsquare = (hdif as f64 / 360.0) * (hdif as f64 / 360.0);
  let ssquare = (sfirst - ssecond) as f64 * ((sfirst - ssecond) as f64) / 10000f64;
  let vsquare = (vfirst - vsecond) as f64 * ((vfirst - vsecond) as f64) / 10000f64;
  let smax = min(sfirst, ssecond);
  let vmax = min(vfirst, vsecond);
  if log {
      println!("{} {} {} {} {} {}", hdif, hsquare, ssquare, vsquare, smax, vmax);
  }
  // When the value is low, the saturation and hue should not matter.
  // When saturation is low, the hue should not matter.
  let retv=if vmax < 20 {
    (vsquare * 10000.0) as i32
  } else if smax < 20 {
    ((vsquare + ssquare) * 10000.0) as i32
  } else {
    ((vsquare + ssquare + 10.0*hsquare) * 10000.0) as i32
  };
  // assert!(vmax <= 100);
  // assert!(smax <= 100);
  // let vmax_q = vmax as f64 / 100.0;
  // let smax_q = smax as f64 / 100.0;
  // let retv = ((vsquare + ssquare * vmax_q + hsquare * vmax_q * smax_q) * 10000.0) as i32;
  if log {
      println!("{}", retv);
  }
  retv
}

fn rgb_to_hsv_f64(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
  let r_norm = r / 255.0;
  let g_norm = g / 255.0;
  let b_norm = b / 255.0;

  let c_max = r_norm.max(g_norm).max(b_norm);
  let c_min = r_norm.min(g_norm).min(b_norm);
  let delta = c_max - c_min;

  let h = if delta == 0.0 {
      0.0
  } else if c_max == r_norm {
      60.0 * (((g_norm - b_norm) / delta) % 6.0)
  } else if c_max == g_norm {
      60.0 * (((b_norm - r_norm) / delta) + 2.0)
  } else {
      60.0 * (((r_norm - g_norm) / delta) + 4.0)
  };

  let s = if c_max == 0.0 {
      0.0
  } else {
      delta / c_max
  };

  let v = c_max;

  (h, s, v)
}
fn rgb_to_hsv(rgb: &Rgb<u8>) -> Rgb<u16> {
  let (h, s, v) = rgb_to_hsv_f64(rgb.0[0] as f64, rgb.0[1] as f64, rgb.0[2] as f64);
  Rgb([h as u16, (s * 100.0) as u16, (v * 100.0) as u16])
}

fn rotate(image: &RgbImage, angle: f32) -> RgbImage {
    // Rotesc o imagine in jurul centrului ei.
    // Centrul este (width/2, height/2).
    // Un punct (x, y) devine (dx, dy) = (x-width/2, y-height/2).
    //
    // There is a real number A and an angle alpha such that dx = A cos alpha,
    // dy = A sin alpha.
    //
    // By rotating with an angle beta, we get the point
    // (A cos(alpha+beta), A sin(alpha+beta))
    //
    // cos(u+v) = cos u cos v - sin u sin v
    // sin(u+v) = sin u cos v + cos u sin v
    //
    // The new point is
    // ( A (cos(alpha) cos(beta) - sin(alpha) sin(beta))
    // , A (sin(alpha) cos(beta) + cos(alpha) sin(beta))
    // )
    // = (dx cos beta - dy sin beta, dy cos beta + dx sin beta)
    //
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();
    let mut new_image = RgbImage::new(image.width(), image.height());
    let width_i32: i32 = image.width().try_into().unwrap();
    let height_i32: i32 = image.height().try_into().unwrap();
    let midx:i32 = width_i32 / 2;
    let midy:i32 = height_i32 / 2;
    for dx in -midx .. width_i32 - midx {
        let x: u32 = (dx + midx).try_into().unwrap();
        let dx_f32 = dx as f32;
        for dy in -midy .. height_i32 - midy {
            let y: u32 = (dy + midy).try_into().unwrap();
            let dy_f32 = dy as f32;
            let old_dx_f32 = dx_f32 * cos_angle - dy_f32 * sin_angle;
            let old_dy_f32 = dy_f32 * cos_angle + dx_f32 * sin_angle;
            let old_x = midx + (old_dx_f32.round() as i32);
            let old_y = midy + (old_dy_f32.round() as i32);
            if 0 <= old_x && old_x < width_i32 && 0 <= old_y && old_y < height_i32 {
                let old_x_u32: u32 = old_x.try_into().unwrap();
                let old_y_u32: u32 = old_y.try_into().unwrap();
                new_image.put_pixel(x, y, *image.get_pixel(old_x_u32, old_y_u32));
            }
        }
    }
    new_image
}

fn blue_line(pixels: &Vec<Rgb<u8>>) -> i32 {
    let mut score = 0_i32;

    for pixel in pixels {
        let r: i32 = pixel.0[0].into();
        let g: i32 = pixel.0[1].into();
        let b: i32 = pixel.0[2].into();
        score += max(0, b - (g + r) / 2_i32);
    }
    score
}
