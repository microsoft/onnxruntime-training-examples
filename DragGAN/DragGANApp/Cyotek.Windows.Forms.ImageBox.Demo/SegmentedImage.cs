//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;
//using System.IO;
//using System.Windows;

//using System.Drawing;
//using System.Drawing.Drawing2D;

//using System.Text.Json;
//using System.Text.Json.Serialization;
//using System.Reflection;

//namespace Cyotek.Windows.Forms
//{
//    public class SegmentedImage
//    {
//        [Serializable]
//        public class ADECategoryInfo
//        {
//            public List<int> color { get; set; }
//            public int id { get; set; }
//            public int isthing { get; set; }
//            public string name { get; set; }
//        }
//        [Serializable]
//        public class ADECategories
//        {
//            public List<ADECategoryInfo> categories { get; set; }
//        }

//        public class RegionInfo
//        {
//            public string CatName = "";
//            public Bitmap CatImage;
//            public Rectangle Rect;

//        }
//        private Image m_image;
//        private Bitmap m_labels_image;
//        private int[,] m_pixel_ids;

//        private Dictionary<int, RegionInfo> m_regions;

//        private Bitmap m_image_filter;

//        public ADECategories m_categories_list;
//        public Dictionary<int, ADECategoryInfo> m_categories;

//        // ----------------------------------------------------------

//        public Size Size()
//        {
//            return m_image.Size;
//        }

//        public Image Image()
//        {
//            return m_image;
//        }

//        public int GetID(int x, int y)
//        {
//            return (m_pixel_ids != null) ? m_pixel_ids[y, x] : m_labels_image.GetPixel(x, y).R;
//        }

//        public void LoadCategories(string filename)
//        {
//            m_categories_list = JsonSerializer.Deserialize<ADECategories>(File.ReadAllText(filename));
//            m_categories = new Dictionary<int, ADECategoryInfo>();
//            foreach(var cat in m_categories_list.categories)
//            {
//                m_categories[cat.id] = cat;
//            }
//        }

//        public void LoadCategories()
//        {
//            var assembly = Assembly.GetExecutingAssembly();
//            var resourceName = "ADE_Categories";

//            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
//            using (StreamReader reader = new StreamReader(stream))
//            {
//                string result = reader.ReadToEnd();

//                m_categories_list = JsonSerializer.Deserialize<ADECategories>(result);
//                m_categories = new Dictionary<int, ADECategoryInfo>();
//                foreach (var cat in m_categories_list.categories)
//                {
//                    m_categories[cat.id] = cat;
//                }
//            }
//        }

//        public void Load(string image_filename)
//        {
//            m_image = System.Drawing.Image.FromFile(image_filename);

//            var width = this.Size().Width;
//            var height = this.Size().Height;

//            m_pixel_ids = new int[height, width];
//            for (var i = 0; i < height; i++)
//            {
//                for (var j = 0; j < width; j++)
//                {
//                    var id = (int)(i / 400) * 100 + (int)(j / 400);

//                    m_pixel_ids[i, j] = id;
//                }
//            }

//            Preprocess();
//        }

//        public void Load2(string image_filename, string labels_filename)
//        {
//            m_image = System.Drawing.Image.FromFile(image_filename);

//            var width = this.Size().Width;
//            var height = this.Size().Height;

//            m_labels_image = (Bitmap)System.Drawing.Image.FromFile(labels_filename);

//            //m_pixel_ids = new int[height, width];
//            //for (var i = 0; i < height; i++)
//            //{
//            //    for (var j = 0; j < width; j++)
//            //    {
//            //        var id = (int)(i / 400) * 100 + (int)(j / 400);

//            //        m_pixel_ids[i, j] = id;
//            //    }
//            //}

//            Preprocess();
//        }


//        public RegionInfo Region(int x, int y)
//        {
//            var id = GetID(x, y); // m_pixel_ids[y, x];

//            var r = m_regions[id];

//            return r;
//        }


//        public Bitmap ImageFilter()
//        {
//            return m_image_filter;
//        }
//        public void Preprocess()
//        {
//            var width = this.Size().Width;
//            var height = this.Size().Height;

//            {
//                m_image_filter = new Bitmap(width, height);
//                var Bmp = m_image_filter;
//                using (Graphics gfx = Graphics.FromImage(Bmp))
//                using (SolidBrush brush = new SolidBrush(Color.FromArgb(200, 255, 255, 255)))
//                {
//                    gfx.FillRectangle(brush, 0, 0, width, height);
//                }
//            }


//            m_regions = new Dictionary<int, RegionInfo>();
//            for (var i = 0; i < height; i++)
//            {
//                for (var j = 0; j < width; j++)
//                {
//                    var id = GetID(j, i);
//                    if (!m_regions.ContainsKey(id))
//                    {

//                        var new_region = new RegionInfo()
//                        {
//                            CatName = (m_categories != null) ? m_categories[id].name : "RG ID " + id.ToString(),
//                            CatImage = new Bitmap(width, height),
//                            Rect = new Rectangle(j, i, 1, 1),
//                        };

//                        var Bmp = new_region.CatImage;
//                        using (Graphics gfx = Graphics.FromImage(Bmp))
//                        using (SolidBrush brush = new SolidBrush(Color.FromArgb(0, 0, 0, 0)))
//                        {
//                            gfx.FillRectangle(brush, 0, 0, width, height);
//                        }

//                        m_regions[id] = new_region;

//                    }

//                    var region = m_regions[id];

//                    //region.CatImage.SetPixel(j, i, Color.Black);
//                    region.CatImage.SetPixel(j, i, ((Bitmap)m_image).GetPixel(j, i));
//                    region.Rect = Rectangle.Union(region.Rect, new Rectangle(j, i, 1, 1));
//               }

//            }
//        }
//        public Bitmap PrepareSelectedImage(List<RegionInfo> regions, bool crop = false)
//        {
//            var width = this.Size().Width;
//            var height = this.Size().Height;

//            var crop_rect = regions[0].Rect;

//            var image = new Bitmap(width, height);
//            var Bmp = image;
//            using (Graphics gfx = Graphics.FromImage(Bmp))
//            using (SolidBrush brush = new SolidBrush(Color.FromArgb(0, 0, 0, 0)))
//            {
//                gfx.FillRectangle(brush, 0, 0, width, height);

//                foreach (var r in regions)
//                {
//                    gfx.DrawImage(r.CatImage, new Rectangle(0,0,width,height));
//                    crop_rect = Rectangle.Union(crop_rect, r.Rect);
//                }
//            }

//            if (crop)
//            {
//                image = image.Clone(crop_rect, Bmp.PixelFormat);
//            }

//            return image;
//        }

//    }
//}

