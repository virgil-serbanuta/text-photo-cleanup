use image::buffer::ConvertBuffer;
use image::GrayImage;
use image::ImageBuffer;
use image::ImageReader;
use image::Luma;
use image::Rgb;
use image::RgbImage;
use std::cmp::max;
use std::cmp::min;
use std::env;
use std::fmt::Debug;

const EDGES_THRESHOLD: i32 = 10;
const EDGES_SURROUNDING_MULTIPLIER: i32 = 4;
const EDGES_SURROUNDING_THRESHOLD_RADIUS: i32 = 10;

pub type BoolImage = ImageBuffer<Luma<u8>, Vec<u8>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
      panic!("args");
    }
    let img = ImageReader::open(args[1].clone());
    let img_rgb:RgbImage = img?.decode()?.into_rgb8();

    let gray: GrayImage = img_rgb.convert();

    let edges: GrayImage = detect_edges(&gray, EDGES_THRESHOLD);
    let bedges: BoolImage = adaptive_threshold(&edges, EDGES_THRESHOLD);
    let selected: BoolImage = select_pixels_adaptive(&gray, &bedges);
    let cleaned: RgbImage = select_pixels(&selected, &img_rgb);
    cleaned.save(args[2].clone())?;

    Ok(())
}

// ******************************************
//                Edges
// ******************************************

fn detect_edges(grayscale: &GrayImage, threshold: i32) -> GrayImage {
  let basic_edges: GrayImage = detect_basic_edges(&grayscale);
  let enhanced_edges: GrayImage = enhance_edges(&basic_edges, threshold);
  trim_edges(enhanced_edges)
}

fn trim_edges(edges: GrayImage) -> GrayImage {
  edges
}

fn get_pixel(image: &GrayImage, x: u32, y: u32) -> i32 {
  match image.get_pixel(y, x) {
    Luma(x) => x[0].into()
  }
}

fn put_pixel<U:TryInto<u8>>(image: &mut GrayImage, x: u32, y: u32, value: U)
    where <U as TryInto<u8>>::Error: Debug
{
  image.put_pixel(y, x, Luma([value.try_into().unwrap()]))
}

fn enhance_edges(edges: &GrayImage, threshold: i32) -> GrayImage {
  let height = edges.height();
  let width = edges.width();
  let mut retv: GrayImage = ImageBuffer::new(width, height);
  for i in 1 .. height - 1 {
    for j in 1 .. width - 1 {
      let eij: i32 = get_pixel(edges, i, j);
      put_pixel(&mut retv, i, j, eij);
      if eij > threshold {
        continue;
      }

      let ep1m1 = eij > get_pixel(edges, i + 1, j - 1);
      let eem1 = eij > get_pixel(edges, i, j - 1);
      let ep1e = eij > get_pixel(edges, i + 1, j);

      let em1p1 = eij > get_pixel(edges, i - 1, j + 1);
      let eep1 = eij > get_pixel(edges, i, j + 1);
      let em1e = eij > get_pixel(edges, i - 1, j);
      if get_pixel(edges, i - 1, j - 1) > threshold || get_pixel(edges, i + 1, j + 1) > threshold {
        // x . .
        // . \ .
        // . . x
        if (ep1m1 || eem1 || ep1e) && (em1p1 || eep1 || em1e) {
          put_pixel(&mut retv, i, j, threshold + 1);
          continue;
        }
      }

      let em1m1 = eij > get_pixel(edges, i - 1, j - 1);
      let ep1p1 = eij > get_pixel(edges, i + 1, j + 1);

      if get_pixel(edges, i + 1, j - 1) > threshold || get_pixel(edges, i - 1, j + 1) > threshold {
        // . . x
        // . / .
        // x . .
        if (em1m1 || em1e || eem1) && (eep1 || ep1e || ep1p1) {
          put_pixel(&mut retv, i, j, threshold + 1);
          continue;
        }
      }
      if get_pixel(edges, i, j - 1) > threshold || get_pixel(edges, i, j + 1) > threshold {
        // . . .
        // x - x
        // . . .
        if (em1m1 || em1e || em1p1) && (ep1m1 || ep1e || ep1p1) {
          put_pixel(&mut retv, i, j, threshold + 1);
          continue;
        }
      }
      if get_pixel(edges, i - 1, j) > threshold || get_pixel(edges, i + 1, j) > threshold {
        // . x .
        // . | .
        // . x .
        if (em1m1 || eem1 || ep1m1) && (em1p1 || eep1 || ep1p1) {
          put_pixel(&mut retv, i, j, threshold + 1);
          continue;
        }
      }
    }
  }
  // return edges;
  return retv;
}

fn detect_basic_edges(grayscale: &GrayImage) -> GrayImage {
  let height = grayscale.height();
  let width = grayscale.width();
  let mut edges: GrayImage = ImageBuffer::new(width, height);
  for x in 1 .. height - 1 {
    for y in 1 .. width - 1 {
      let a1 = get_pixel(grayscale, x - 1, y - 1) - get_pixel(grayscale, x + 1, y - 1);
      let a2 = get_pixel(grayscale, x - 1, y) - get_pixel(grayscale, x + 1, y);
      let a3 = get_pixel(grayscale, x - 1, y + 1) - get_pixel(grayscale, x + 1, y + 1);

      let a4 = get_pixel(grayscale, x - 1, y - 1) - get_pixel(grayscale, x - 1, y + 1);
      let a5 = get_pixel(grayscale, x, y - 1) - get_pixel(grayscale, x, y + 1);
      let a6 = get_pixel(grayscale, x + 1, y - 1) - get_pixel(grayscale, x + 1, y + 1);

      let a7 = get_pixel(grayscale, x - 1, y - 1) - get_pixel(grayscale, x + 1, y + 1);
      let a8 = get_pixel(grayscale, x - 1, y + 1) - get_pixel(grayscale, x + 1, y - 1);

      let g = (a1.abs() + a2.abs() + a3.abs()
                      + a4.abs() + a5.abs() + a6.abs()
                      + a7.abs() + a8.abs()
              ) / 8;
      assert!(0 <= g && g <= 255);
      put_pixel(&mut edges, x, y, g);
    }
  }
  return edges;
}

// ******************************************
//                Threshold
// ******************************************

fn adaptive_threshold(edges: &GrayImage, limit: i32) -> BoolImage {
  let height = edges.height();
  let width = edges.width();
  let mut surrounding_edge : GrayImage =
      ImageBuffer::new
          ( width / (EDGES_SURROUNDING_THRESHOLD_RADIUS as u32) + 1u32
          , height / (EDGES_SURROUNDING_THRESHOLD_RADIUS as u32) + 1u32
          );
  for i in 0 .. height {
    let idiv = i / (EDGES_SURROUNDING_THRESHOLD_RADIUS as u32);
    for j in 0 .. width {
      let jdiv = j / (EDGES_SURROUNDING_THRESHOLD_RADIUS as u32);
      let existing = get_pixel(&edges, i, j);
      if get_pixel(&surrounding_edge, idiv, jdiv) < existing {
        put_pixel(&mut surrounding_edge, idiv, jdiv, existing);
      }
    }
  }
  let mut retv : BoolImage = ImageBuffer::new(width, height);
  for i in 0 .. height {
    for j in 0 .. width {
      let edges_ij = get_pixel(&edges, i, j);
      if edges_ij < limit {
        continue;
      }
      let idiv = i / (EDGES_SURROUNDING_THRESHOLD_RADIUS as u32);
      let jdiv = j / (EDGES_SURROUNDING_THRESHOLD_RADIUS as u32);
      if EDGES_SURROUNDING_MULTIPLIER * edges_ij < get_pixel(&surrounding_edge, idiv, jdiv) {
        continue;
      }
      put_pixel(&mut retv, i, j, 1);
    }
  }
  return retv;
}

// ******************************************
//                Select pixels
// ******************************************

const RADIUS: i32 = 10;

fn select_pixels_adaptive(grayscale: &GrayImage, edges: &BoolImage) -> BoolImage {
  let height = grayscale.height();
  let width = grayscale.width();
  let mut selected : BoolImage = ImageBuffer::new(width, height);
  for x in 0 .. height {
    let minx: u32 = max(x as i32 - RADIUS, 0).try_into().unwrap();
    let maxx: u32 = min(x as i32 + RADIUS, height as i32 - 1).try_into().unwrap();
    for y in 1 .. width - 1 {
      let miny: u32 = max(y as i32 - RADIUS, 0).try_into().unwrap();
      let maxy: u32 = min(y as i32 + RADIUS, width as i32 - 1).try_into().unwrap();

      let mut gt = 0;
      let mut edge = false;
      let for_gt = get_pixel(&grayscale, x, y) * 12 / 10;
      for nx in minx ..= maxx {
        for ny in miny ..= maxy {
          let ng = get_pixel(grayscale, nx, ny);
          if ng > for_gt {
            gt += 1;
          }
          edge = edge || get_pixel(edges, nx, ny) != 0;
        }
      }
      if edge && gt > 2*RADIUS {
        put_pixel(&mut selected, x, y, 1);
      }
    }
  }
  return selected;
}

fn select_pixels(selected: &BoolImage, original: &RgbImage) -> RgbImage {
  let height = original.height();
  let width = original.width();
  let mut result : RgbImage = ImageBuffer::new(width, height);
  for x in 0 .. height {
    for y in 0 .. width {
      if get_pixel(selected, x, y) != 0 {
        result.put_pixel(y, x, original.get_pixel(y, x).clone());
      } else {
        result.put_pixel(y, x, Rgb([255_u8, 255_u8, 255_u8]))
      }
    }
  }
  return result;
}
