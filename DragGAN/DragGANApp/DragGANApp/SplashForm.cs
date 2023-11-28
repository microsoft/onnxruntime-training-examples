using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DragGANApp
{
    public partial class SplashForm : Form
    {
        private Timer _timer;
        private int _cTick = 5;
        public SplashForm()
        {
            InitializeComponent();

            _timer = new Timer() { Interval = 1000, Enabled = true };
            _timer.Tick += CTick;
        }

        private void CTick(object sender, EventArgs e)
        {
            if (--_cTick < 0)
            {
                _timer.Enabled = false;
                Close();

            }
        }
    }
}
