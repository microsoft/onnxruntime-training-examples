// Cyotek ImageBox
// http://cyotek.com/blog/tag/imagebox

// Copyright (c) 2021 Cyotek Ltd.

// This work is licensed under the MIT License.
// See LICENSE.TXT for the full text

// Found this code useful?
// https://www.cyotek.com/contribute

using System;
using System.Windows.Forms;

namespace Cyotek.Windows.Forms.Demo
{
  internal partial class PanStylesDemoForm : BaseForm
  {
    #region Public Constructors

    public PanStylesDemoForm()
    {
      this.InitializeComponent();
    }

    #endregion Public Constructors

    #region Protected Methods

    protected override void OnLoad(EventArgs e)
    {
      base.OnLoad(e);

      propertyGrid.BrowsableProperties = new[]
      {
        nameof(ImageBox.AllowFreePan),
        nameof(ImageBox.InvertMouse),
        nameof(ImageBox.PanMode)
      };

      propertyGrid.SelectedObject = imageBox;
    }

    #endregion Protected Methods

    #region Private Methods

    private void aboutToolStripMenuItem_Click(object sender, EventArgs e)
    {
      AboutDialog.ShowAboutDialog();
    }

    private void closeToolStripMenuItem_Click(object sender, EventArgs e)
    {
      this.Close();
    }

    private void imageBox_PanEnd(object sender, EventArgs e)
    {
      eventsListBox.AddEvent((Control)sender, nameof(ImageBox.PanEnd));
    }

    private void imageBox_PanStart(object sender, EventArgs e)
    {
      eventsListBox.AddEvent((Control)sender, nameof(ImageBox.PanStart));
    }

    #endregion Private Methods
  }
}
