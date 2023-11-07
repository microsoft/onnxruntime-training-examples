using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DragGANApp.Tiles
{
    public partial class StyleGANSample : UserControl
    {
        public enum TileTypes { Picture, Text }

        public TileTypes Type { get; set; }

        private bool m_selected = false;
        public bool Selected
        {
            get
            {
                return this.m_selected;
            }
            set
            {
                m_selected = value;

                if (m_selected)
                {
                    panelClient.Padding = new Padding(5);
                    panelClient.BackColor = Color.LawnGreen;
                }
                else
                {
                    panelClient.Padding = new Padding(0);
                    panelClient.BackColor = SystemColors.Control;
                }
            }
        }

        public event StyleGANItemDelegate OnItemClick;
        public event StyleGANItemDelegate OnItemDoubleClick;

        //public int Seed;
        public DragGAN.Info Info;

        public string Title
        {
            get
            {
                return labelTitle.Text;
            }
            set
            {
                labelTitle.Text = value;
            }
        }


        // ---------------------------------------------------------------
        public StyleGANSample()
        {
            InitializeComponent();

            this.BackColor = Color.Transparent;
            panelTop.BackColor = Color.Transparent;
            panelBottom.BackColor = Color.Transparent;
            //labelTitle.BackColor = Color.Transparent;
            //label1.BackColor = Color.Transparent;

            Selected = false;
        }


        public void Init(TileTypes type)
        {
            this.Type = type;

            switch (this.Type)
            {
                case TileTypes.Picture:
                    this.richTextBox.Visible = false;
                    this.pictureBox.Dock = DockStyle.Fill;
                    break;

                case TileTypes.Text:
                    this.pictureBox.Visible = false;
                    this.richTextBox.Dock = DockStyle.Fill;
                    break;
            }
        }

        public void SetPicture(Image im)
        {
            this.pictureBox.Image = im;
        }

        public bool VisibleBottom
        {
            get
            {
                return this.panelBottom.Visible;
            }
            set
            {
                this.panelBottom.Visible = value;
            }
        }


        private void pictureBox_Click(object sender, EventArgs e)
        {
            RaiseOnClick();
        }

        private void labelTitle_Click(object sender, EventArgs e)
        {
            RaiseOnClick();
        }

        private void labelTitle_DoubleClick(object sender, EventArgs e)
        {
            RaiseOnDoubleClick();
        }

        private void panelClient_Click(object sender, EventArgs e)
        {
            RaiseOnClick();
        }

        private void pictureBox_DoubleClick(object sender, EventArgs e)
        {
            RaiseOnDoubleClick();
        }

        public void RaiseOnClick()
        {
            OnItemClick?.Invoke(this);
        }

        public void RaiseOnDoubleClick()
        {
            OnItemDoubleClick?.Invoke(this);
        }


    }
}
