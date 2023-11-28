using Cyotek.Windows.Forms;
using DragGANApp.Tiles;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using static DragGANApp.DragGAN;

namespace DragGANApp
{
    public partial class MainForm : Form
    {
        public DragGAN.Info m_current;
        public Stack<DragGAN.Info> m_history = new Stack<DragGAN.Info>();
        public Stack<DragGAN.Info> m_redo_buffer = new Stack<DragGAN.Info>();

        public Bitmap m_original_image;
        public Bitmap m_edit_image;
        public Tensor<float> m_edit_latent;

        public enum MouseSelectModes { NoPoins, FirstPoint, SecondPoint }
        public MouseSelectModes SelectMode = MouseSelectModes.NoPoins;

        public Point FirstPoint = new Point(-1, -1);
        public Point SecondPoint = new Point(-1, -1);

        public int HandleSize = 5;

        //public StyleGAN m_sg;
        public DragGAN m_dg;

        System.Windows.Forms.Cursor BitMapCursor;
        string m_dragdrop_temp_fn = "DragDropTemp.png";

        bool m_optimizing = false;
        Thread m_optimization_thread = null;

        public Misc.Options m_options;

        // --------------------------------------------------
        public MainForm()
        {
            InitializeComponent();
            DoubleBuffered = true;

            m_options = Misc.LoadAppOptions();

            // update the model folder
            var CurrentExecutable = Environment.GetCommandLineArgs()[0];
            var CurrentExecutableFolder = Path.GetDirectoryName(CurrentExecutable);
            var NewModelPath = new FileInfo(m_options.Model.Replace("$", CurrentExecutableFolder)).FullName;
            m_options.Model = NewModelPath;

            this.BackColor = Color.FromArgb(255, m_options.BGColor.R, m_options.BGColor.G, m_options.BGColor.B);

            TurnBColor(this);

            imageBoxEdit.GiveFeedback += new GiveFeedbackEventHandler(ImageBoxEdit_GiveFeedback);
            imageBoxView.GiveFeedback += new GiveFeedbackEventHandler(ImageBoxView_GiveFeedback);
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (keyData == (Keys.Alt | Keys.Menu))
            {
                if (!this.menuStripMain.Visible)
                {
                    this.menuStripMain.Visible = true;
                }
                else
                {
                    this.menuStripMain.Visible = false;
                }
                return true;
            }
            return base.ProcessCmdKey(ref msg, keyData);
        }


        void TurnBColor(Control c)
        {
            try
            {
                c.BackColor = this.BackColor;
                foreach (Control child in c.Controls)
                {
                    TurnBColor(child);
                }
            }
            catch (Exception)
            {
                //Console.WriteLine($"Got an Exception for control {c.Name}\n{e.ToString()}");
                Console.WriteLine($"Got an Exception for control {c.Name}");
            }
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            var splash_form = new SplashForm();
            splash_form.Show();
            System.Windows.Forms.Application.DoEvents();

            m_dg = new DragGAN();
            m_dg.Init(m_options.Model, m_options.Iterations);

            styleGANSamplesContainer1.OnItemClick += onItemClick;
            styleGANSamplesContainer1.OnItemDoubleClick += onItemDoubleClick;
            styleGANSamplesContainer1.Init(m_dg, m_options.InitialSeed);
        }

        private void MainForm_Shown(object sender, EventArgs e)
        {

        }


        private void imageBoxEdit_MouseClick(object sender, MouseEventArgs e)
        {
            Console.WriteLine($"MOUSE CLICK: {e.X}, {e.Y}");

            if (e.Button == MouseButtons.Right)
            {
                if (imageBoxEdit.IsPointInImage(e.Location))
                {
                    MaybeStopOptimization();
                    // add a new landmark
                    var p = imageBoxEdit.PointToImage(e.Location);

                    ProcessClick(p);
                }
            }
        }

        void ProcessClick(Point p)
        {
            switch (SelectMode)
            {
                case MouseSelectModes.NoPoins:
                case MouseSelectModes.SecondPoint:
                    {
                        SelectMode = MouseSelectModes.FirstPoint;
                        FirstPoint = p;
                    }
                    break;
                case MouseSelectModes.FirstPoint:
                    {
                        SelectMode = MouseSelectModes.SecondPoint;
                        SecondPoint = p;
                    }
                    break;
            }

            // update and draw the editd image
            {
                m_edit_image = (Bitmap)m_original_image.Clone();
                using (Graphics g = Graphics.FromImage(m_edit_image))
                {
                    if (SelectMode == MouseSelectModes.FirstPoint || SelectMode == MouseSelectModes.SecondPoint)
                    {
                        using (Brush brush = new SolidBrush(Color.Red))
                        {
                            var point = FirstPoint;
                            g.FillEllipse(brush, point.X - HandleSize, point.Y - HandleSize, HandleSize * 2, HandleSize * 2);
                        }
                    }

                    if (SelectMode == MouseSelectModes.SecondPoint)
                    {
                        using (Brush brush = new SolidBrush(Color.Blue))
                        {
                            var point = SecondPoint;
                            g.FillEllipse(brush, point.X - HandleSize, point.Y - HandleSize, HandleSize * 2, HandleSize * 2);
                        }
                    }

                }
                // update the image on screen
                UpdateImage(m_edit_image);
            }
        }

        public void UpdateImage(Bitmap new_image)
        {
            imageBoxEdit.Image = new_image;
            imageBoxView.Image = new_image;
        }

        private void runToolStripMenuItem_Click(object sender, EventArgs e)
        {
            RunOptimization();
        }

        public void UpdateNewCurrent(DragGAN.Info new_current)
        {
            if (m_current != null)
            {
                m_history.Push(m_current);
                m_redo_buffer.Clear();
            }
            m_current = new_current;
        }

        private void undoonelyOnceToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (m_history.Count == 0)
            {
                MessageBox.Show("No Previous Images in history");
                return;
            }

            m_redo_buffer.Push(m_current);

            var prev = m_history.Pop();
            
            m_current = prev;
            LoadImage(false);
        }

        private void redoToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (m_redo_buffer.Count == 0)
            {
                MessageBox.Show("No more redo steps");
                return;
            }

            var redo = m_redo_buffer.Pop();
            m_history.Push(m_current);

            m_current = redo;
            LoadImage(false);
        }

        public void onItemClick(StyleGANSample sample)
        {
        }

        public void onItemDoubleClick(StyleGANSample sample)
        {
            MaybeStopOptimization();
            UpdateNewCurrent(sample.Info);
            LoadImage();
        }

        public void LoadImage(bool zoom_to_fit = true)
        {
            m_original_image = (Bitmap)m_current.Image.Clone();

            m_edit_image = (Bitmap)m_original_image.Clone();

            UpdateImage(m_edit_image);

            if (zoom_to_fit)
                imageBoxEdit.ZoomToFit();
            imageBoxView.ZoomToFit();
        }

        private void imageBoxView_SizeChanged(object sender, EventArgs e)
        {
            imageBoxView.ZoomToFit();
        }

        private void imageBoxEdit_MouseDown(object sender, MouseEventArgs e)
        {
            if ((Control.ModifierKeys & Keys.Shift) != 0)
            {
                var image_to_paste = m_edit_image;
                image_to_paste.Save(m_dragdrop_temp_fn);

                this.BitMapCursor = new System.Windows.Forms.Cursor((DragGAN.ResizeImage(image_to_paste, new Size(100, 100)).GetHicon()));

                DataObject obj = new DataObject();
                var list = new System.Collections.Specialized.StringCollection() { Path.GetFullPath(m_dragdrop_temp_fn) };
                obj.SetFileDropList(list);

                imageBoxEdit.DoDragDrop(obj, DragDropEffects.All);
            }
        }

        private void imageBoxView_MouseDown(object sender, MouseEventArgs e)
        {
            imageBoxEdit_MouseDown(sender, e);
        }

        private void ImageBoxEdit_GiveFeedback(object sender, GiveFeedbackEventArgs e)
        {
            //Deactivate the default cursor
            e.UseDefaultCursors = false;

            //Use the cursor created from the bitmap
            System.Windows.Forms.Cursor.Current = this.BitMapCursor;
        }

        private void ImageBoxView_GiveFeedback(object sender, GiveFeedbackEventArgs e)
        {
            //Deactivate the default cursor
            e.UseDefaultCursors = false;

            //Use the cursor created from the bitmap
            System.Windows.Forms.Cursor.Current = this.BitMapCursor;
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            System.Windows.Forms.Application.Exit();
        }

        private void stopOptimizationToolStripMenuItem_Click(object sender, EventArgs e)
        {
            MaybeStopOptimization();
        }

        private void RunOptimization()
        {
            if (FirstPoint.X < 0 || SecondPoint.X < 0)
            {
                MessageBox.Show("Please select valid handles using the mouse right-click button before trying to optimize the image", "Warning");
                return;
            }
            if (m_optimizing)
            {
                MessageBox.Show("Seems line the optimization process is still running - this should not happen. please inform rgal@microsoft.com", "PROBLEM!");
                return;
            }
            try
            {
                m_optimizing = true;

                var handles = new List<ImageCoord>() { new ImageCoord(FirstPoint.X, FirstPoint.Y) };
                var targets = new List<ImageCoord>() { new ImageCoord(SecondPoint.X, SecondPoint.Y) };

                DragGAN.OptimizationParameters opt_params = new DragGAN.OptimizationParameters()
                {
                    Handles = handles,
                    Targets = targets,
                    Latent = m_current.Latent,
                    IntermidiateImagesAction = (info) =>
                    {
                        this.Invoke((MethodInvoker)delegate {
                            UpdateImage(info.Image);
                            imageBoxEdit.Text = $"Optimization in progress: {info.CurrentIteration + 1}/{info.TotalIterations}";
                        });
                        //imageBoxEdit.Image = img;
                        System.Windows.Forms.Application.DoEvents();
                    },
                    OptimizationEndedAction = this.OptimizationEnded,
                };

                imageBoxEdit.GridColor = Color.Moccasin; // Color.RosyBrown;
                imageBoxView.GridColor = Color.Moccasin; //Color.RosyBrown;


                m_optimization_thread = new Thread(new ParameterizedThreadStart(m_dg.RunInThread));
                m_optimization_thread.Start(opt_params);
            }
            catch(Exception ex) 
            {
                MessageBox.Show(ex.ToString(), "EXCEPTION");
            }
            finally
            {
                //m_optimizing = false;
            }
        }

        private void OptimizationEnded()
        {
            try
            {
                UpdateNewCurrent(m_dg.LatestResult);
                LoadImage(false);

                this.Invoke((MethodInvoker)delegate {
                    imageBoxEdit.GridColor = Color.Gainsboro;
                    imageBoxView.GridColor = Color.Gainsboro;
                    imageBoxEdit.Text = "";
                });
            }
            catch (Exception ex) 
            {
                MessageBox.Show(ex.ToString(), "EXCEPTION");
            }
            finally
            {
                m_optimizing = false;
            }
        }

        private void MaybeStopOptimization()
        {
            if (m_optimizing)
            {
                m_dg.OptimizationShouldStop = true;
                bool terminated = false;
                while (!terminated)
                {
                    System.Windows.Forms.Application.DoEvents();
                    terminated = m_optimization_thread.Join(100);
                }
                m_optimization_thread = null;

            }
            else
            {
                // do nothing
            }
        }

        private void buttonOptimize_Click(object sender, EventArgs e)
        {
            RunOptimization();
        }

        private void buttonStop_Click(object sender, EventArgs e)
        {
            MaybeStopOptimization();
        }

        private void buttonUndo_Click(object sender, EventArgs e)
        {
            undoonelyOnceToolStripMenuItem_Click(null, null);
        }

        private void buttonRedo_Click(object sender, EventArgs e)
        {
            redoToolStripMenuItem_Click(null, null);
        }

        private void MainForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            m_dg.Shutdown();
        }
    }
}
