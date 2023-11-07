using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DragGANApp.Tiles
{
    public partial class StyleGANSamplesContainer : UserControl
    {
        DragGAN m_sg;

        int m_batch_size = 10;
        public int BatchSize
        {
            get
            {
                return m_batch_size;
            }
            set 
            {
                if (m_sg != null)
                {
                    m_batch_size = value;
                    RePopulateContainer();
                }
            }
        }

        public event StyleGANItemDelegate OnItemClick;
        public event StyleGANItemDelegate OnItemDoubleClick;

        public int CurrentSeed { get { return (int)numericUpDownSeed.Value; } }
        bool m_loading = true;

        public int GroupHeight { get; set; }

        public int[] AllowedHeights = new int[] { 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1200 };
        public int GroupHeightIndex = 2;


        [DllImport("user32.dll")]
        private static extern long LockWindowUpdate(IntPtr Handle);
        // ------------------------------------------------------
        public StyleGANSamplesContainer()
        {
            InitializeComponent();

            this.flowLayoutPanelItems.MouseWheel += new System.Windows.Forms.MouseEventHandler(this.form_MouseWheel);
        }


        public void Init(DragGAN sg, int initial_seed = 0)
        {
            m_sg = sg;
            //numericUpDownSeed.Value = 10;
            numericUpDownSeed.Value = initial_seed;
            BatchSize = 10;
        }

        public Form ParentForm
        {
            get { return GetParentForm(this.Parent); }
        }

        private Form GetParentForm(Control parent)
        {
            Form form = parent as Form;
            if (form != null)
            {
                return form;
            }
            if (parent != null)
            {
                // Walk up the control hierarchy
                return GetParentForm(parent.Parent);
            }
            return null; // Control is not on a Form
        }

        private void SuspendLayoutUpdate()
        {
            this.SuspendLayout();
            this.flowLayoutPanelItems.SuspendLayout();
        }

        private void ResumeLayoutUpdate()
        {
            this.flowLayoutPanelItems.ResumeLayout(true);
            this.ResumeLayout();
        }


        public StyleGANSample CreateItemControl(int seed, string title)
        {
            var item_control = new StyleGANSample();

            item_control.Init(StyleGANSample.TileTypes.Picture);

            var info = m_sg.GetImage(seed);

            item_control.SetPicture(info.Image);
            item_control.Title = title;
            item_control.VisibleBottom = false;
            item_control.OnItemClick += onItemClick;
            item_control.OnItemDoubleClick += onItemDoubleClick;

            item_control.Info = info;

            return item_control;
        }

        public void RePopulateContainer()
        {
            try
            {
                // Lock Window...
                LockWindowUpdate(ParentForm.Handle);
                // Perform your painting / updates...

                SuspendLayoutUpdate();
                flowLayoutPanelItems.Controls.Clear();
                int item_index = 0;

                for (int seed = CurrentSeed; seed < CurrentSeed + BatchSize; seed++)
                {
                    var item_control = CreateItemControl(seed, $"Seed: {seed}");
                    flowLayoutPanelItems.Controls.Add(item_control);
                    item_index++;
                }
                SetNewHeight();
                ResumeLayoutUpdate();
            }
            finally
            {
                // Release the lock...
                LockWindowUpdate(IntPtr.Zero);
            }

        }

        public void onItemClick(StyleGANSample c)
        {
            OnItemClick?.Invoke(c);
            ////this.OnTileClick?.Invoke(this, tile);

            //// prepare the image to view
            //var item = c.Item;
            //var image = (Image)item.Capture.Screen.Clone();

            //// draw rectangle around the image
            //Pen skyBluePen = new Pen(Brushes.DeepSkyBlue);
            //skyBluePen.Width = 20.0F;
            //skyBluePen.Color = Color.FromArgb(128, skyBluePen.Color); // make the color transparent
            //skyBluePen.LineJoin = System.Drawing.Drawing2D.LineJoin.Bevel;

            //if (item.Type == DBItem.Types.AllScreen)
            //{
            //    imageBoxScreen.Image = image;
            //    imageBoxScreen.ZoomToFit();
            //    return;
            //}

            //using (Graphics g = Graphics.FromImage(image))
            //{
            //    Pen pen = new Pen(Color.Yellow, 20);
            //    g.DrawRectangle(skyBluePen, item.Region.Rect);
            //}


            //imageBoxScreen.Image = image;
            //imageBoxScreen.ZoomToFit();

            //var r = item.Region.Rect;
            //imageBoxScreen.ZoomToRegion(r.X, r.Y, r.Width, r.Height);

            //imageBoxScreen.ZoomOut();
            //imageBoxScreen.ZoomOut();
            //imageBoxScreen.ZoomOut();
        }

        public void onItemDoubleClick(StyleGANSample c)
        {
            OnItemDoubleClick?.Invoke(c);
            ////this.OnTileDoubleClick?.Invoke(this, tile);
            //Console.WriteLine("DD " + c.Title);
            //FilterItem(c.Item);
        }

        private void buttonNext_Click(object sender, EventArgs e)
        {
            try 
            {
                numericUpDownSeed.Value = numericUpDownSeed.Value + numericUpDownSeed.Increment;
            }
            catch 
            {
                numericUpDownSeed.Value = 30000;
            }
            RePopulateContainer();
        }

        private void buttonPrevious_Click(object sender, EventArgs e)
        {
            try
            {
                numericUpDownSeed.Value = numericUpDownSeed.Value - numericUpDownSeed.Increment;
            }
            catch
            {
                numericUpDownSeed.Value = 0;
            }
            RePopulateContainer();
        }

        private void buttonGoto_Click(object sender, EventArgs e)
        {
            RePopulateContainer();
        }

        public void ZoomIn()
        {
            this.GroupHeightIndex += 1;
            if (this.GroupHeightIndex >= this.AllowedHeights.Length)
                this.GroupHeightIndex = this.AllowedHeights.Length - 1;
            this.SetNewHeight();
        }

        public void ZoomOut()
        {
            this.GroupHeightIndex -= 1;
            if (this.GroupHeightIndex < 0)
                this.GroupHeightIndex = 0;
            this.SetNewHeight();
        }

        private void SetNewHeight()
        {
            //this.flowLayoutPanel.Visible = false;
            try
            {
                // Lock Window...
                LockWindowUpdate(ParentForm.Handle);
                // Perform your painting / updates...

                this.SuspendLayout();
                this.flowLayoutPanelItems.SuspendLayout();
                int size = this.AllowedHeights[this.GroupHeightIndex];
                foreach (StyleGANSample c in flowLayoutPanelItems.Controls)
                {
                    c.Size = new Size(size, size);

                    //c.SetNewHeight(this.AllowedHeights[this.GroupHeightIndex]);
                }
                this.flowLayoutPanelItems.ResumeLayout();
                this.ResumeLayout(true);
            }
            finally
            {
                // Release the lock...
                LockWindowUpdate(IntPtr.Zero);
            }

            //this.SuspendLayout();
            //this.flowLayoutPanelItems.SuspendLayout();
            //int size = this.AllowedHeights[this.GroupHeightIndex];
            //foreach (StyleGANSample c in flowLayoutPanelItems.Controls)
            //{
            //    c.Size = new Size(size, size);

            //    //c.SetNewHeight(this.AllowedHeights[this.GroupHeightIndex]);
            //}
            //this.flowLayoutPanelItems.ResumeLayout();
            //this.ResumeLayout(true);
        }

        private void form_MouseWheel(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            // Update the drawing based upon the mouse wheel scrolling.

            if (Control.ModifierKeys == Keys.Control)
            {
                var scale = (int)(e.Delta * SystemInformation.MouseWheelScrollLines / 120);
                if (scale > 0)
                {
                    this.ZoomIn();
                    //this.GroupHeightIndex += 1;
                    //if (this.GroupHeightIndex >= this.AllowedHeights.Length)
                    //    this.GroupHeightIndex = this.AllowedHeights.Length - 1;
                }
                else
                {
                    this.ZoomOut();
                    //this.GroupHeightIndex -= 1;
                    //if (this.GroupHeightIndex < 0)
                    //    this.GroupHeightIndex = 0;
                }
                //this.SetNewHeight();
            }
        }

    }
}
