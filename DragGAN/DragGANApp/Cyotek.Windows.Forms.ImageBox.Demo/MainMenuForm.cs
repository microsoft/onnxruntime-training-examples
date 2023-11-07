// Cyotek ImageBox
// http://cyotek.com/blog/tag/imagebox

// Copyright (c) 2010-2021 Cyotek Ltd.

// This work is licensed under the MIT License.
// See LICENSE.TXT for the full text

// Found this code useful?
// https://www.cyotek.com/contribute

using System;
using System.Windows.Forms;

namespace Cyotek.Windows.Forms.Demo
{
  internal partial class MainMenuForm : AboutDialog
  {
    #region Constructors

    public MainMenuForm()
    {
      this.InitializeComponent();
    }

    #endregion

    #region Methods

    protected override void OnLoad(EventArgs e)
    {
      TabPage demoPage;

      base.OnLoad(e);

      demoPage = new TabPage
                 {
                   UseVisualStyleBackColor = true,
                   Padding = new Padding(9),
                   Text = "Demonstrations"
                 };

      demoGroupBox.Dock = DockStyle.Fill;
      demoPage.Controls.Add(demoGroupBox);

      this.TabControl.TabPages.Insert(0, demoPage);
      this.TabControl.SelectedTab = demoPage;

      this.Text = "Cyotek ImageBox Control for Windows Forms";
    }

    /// <summary>
    /// Raises the <see cref="E:System.Windows.Forms.Form.Shown"/> event.
    /// </summary>
    /// <param name="e">A <see cref="T:System.EventArgs"/> that contains the event data. </param>
    protected override void OnShown(EventArgs e)
    {
      base.OnShown(e);

      imageBoxDemoButton.Focus();
    }

    private void animatedGifDemoButton_Click(object sender, EventArgs e)
    {
      this.ShowDemo<AnimatedGifDemoForm>();
    }

    private void dragTestDemoButton_Click(object sender, EventArgs e)
    {
      this.ShowDemo<DragTestForm>();
    }

    private void imageBoxDemoButton_Click(object sender, EventArgs e)
    {
      this.ShowDemo<GeneralDemoForm>();
    }

    private void minimapDemoButton_Click(object sender, EventArgs e)
    {
      this.ShowDemo<MiniMapDemoForm>();
    }

    private void panDemoButton_Click(object sender, EventArgs e)
    {
      this.ShowDemo<PanStylesDemoForm>();
    }

    private void pixelGridDemoButton_Click(object sender, EventArgs e)
    {
      this.ShowDemo<PixelGridForm>();
    }

    private void resizableSelectionDemoButton_Click(object sender, EventArgs e)
    {
      this.ShowDemo<ResizableSelectionDemoForm>();
    }

    private void scaledAdornmentsDemoButton_Click(object sender, EventArgs e)
    {
      this.ShowDemo<ScaledAdornmentsDemoForm>();
    }

    private void ShowDemo<T>()
      where T : Form, new()
    {
      Cursor.Current = Cursors.WaitCursor;

      using (Form form = new T())
      {
        form.ShowDialog(this);
      }
    }

    private void sizeModeDemoButton_Click(object sender, EventArgs e)
    {
      this.ShowDemo<SizeModeDemoForm>();
    }

    private void switchImageDuringZoomDemoButton_Click(object sender, EventArgs e)
    {
      this.ShowDemo<SwitchImageDuringZoomDemoForm>();
    }

    private void textDemoButton_Click(object sender, EventArgs e)
    {
      this.ShowDemo<TextDemoForm>();
    }

    private void virtualModeDemoButton_Click(object sender, EventArgs e)
    {
      this.ShowDemo<VirtualModeDemonstrationForm>();
    }

    #endregion
  }
}
