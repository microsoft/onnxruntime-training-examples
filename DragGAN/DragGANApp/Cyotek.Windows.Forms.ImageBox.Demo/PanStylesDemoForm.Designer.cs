namespace Cyotek.Windows.Forms.Demo
{
  partial class PanStylesDemoForm
  {
    /// <summary>
    /// Required designer variable.
    /// </summary>
    private System.ComponentModel.IContainer components = null;

    /// <summary>
    /// Clean up any resources being used.
    /// </summary>
    /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
    protected override void Dispose(bool disposing)
    {
      if (disposing && (components != null))
      {
        components.Dispose();
      }
      base.Dispose(disposing);
    }

    #region Windows Form Designer generated code

    /// <summary>
    /// Required method for Designer support - do not modify
    /// the contents of this method with the code editor.
    /// </summary>
    private void InitializeComponent()
    {
      System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(PanStylesDemoForm));
      this.splitContainer = new System.Windows.Forms.SplitContainer();
      this.imageBox = new Cyotek.Windows.Forms.ImageBox();
      this.propertyGrid = new Cyotek.Windows.Forms.Demo.FilteredPropertyGrid();
      this.demoLabel = new System.Windows.Forms.Label();
      this.eventsListBox = new Cyotek.Windows.Forms.Demo.EventsListBox();
      this.menuStrip = new System.Windows.Forms.MenuStrip();
      this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
      this.closeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
      this.helpToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
      this.aboutToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
      this.statusStrip = new System.Windows.Forms.StatusStrip();
      this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
      this.positionToolStripStatusLabel = new System.Windows.Forms.ToolStripStatusLabel();
      this.splitContainer.Panel1.SuspendLayout();
      this.splitContainer.Panel2.SuspendLayout();
      this.splitContainer.SuspendLayout();
      this.menuStrip.SuspendLayout();
      this.statusStrip.SuspendLayout();
      this.SuspendLayout();
      // 
      // splitContainer
      // 
      this.splitContainer.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
      this.splitContainer.FixedPanel = System.Windows.Forms.FixedPanel.Panel2;
      this.splitContainer.Location = new System.Drawing.Point(10, 23);
      this.splitContainer.Name = "splitContainer";
      // 
      // splitContainer.Panel1
      // 
      this.splitContainer.Panel1.Controls.Add(this.imageBox);
      // 
      // splitContainer.Panel2
      // 
      this.splitContainer.Panel2.Controls.Add(this.propertyGrid);
      this.splitContainer.Panel2.Controls.Add(this.demoLabel);
      this.splitContainer.Panel2.Controls.Add(this.eventsListBox);
      this.splitContainer.Size = new System.Drawing.Size(764, 556);
      this.splitContainer.SplitterDistance = 497;
      this.splitContainer.TabIndex = 1;
      // 
      // imageBox
      // 
      this.imageBox.Dock = System.Windows.Forms.DockStyle.Fill;
      this.imageBox.Image = global::Cyotek.Windows.Forms.Demo.Properties.Resources.Sample;
      this.imageBox.Location = new System.Drawing.Point(0, 0);
      this.imageBox.Name = "imageBox";
      this.imageBox.Size = new System.Drawing.Size(497, 556);
      this.imageBox.TabIndex = 0;
      this.imageBox.PanEnd += new System.EventHandler(this.imageBox_PanEnd);
      this.imageBox.PanStart += new System.EventHandler(this.imageBox_PanStart);
      // 
      // propertyGrid
      // 
      this.propertyGrid.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
      this.propertyGrid.CommandsVisibleIfAvailable = false;
      this.propertyGrid.HelpVisible = false;
      this.propertyGrid.Location = new System.Drawing.Point(3, 352);
      this.propertyGrid.Name = "propertyGrid";
      this.propertyGrid.Size = new System.Drawing.Size(260, 90);
      this.propertyGrid.TabIndex = 1;
      this.propertyGrid.ToolbarVisible = false;
      // 
      // demoLabel
      // 
      this.demoLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
      this.demoLabel.AutoEllipsis = true;
      this.demoLabel.BackColor = System.Drawing.SystemColors.Info;
      this.demoLabel.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
      this.demoLabel.ForeColor = System.Drawing.SystemColors.InfoText;
      this.demoLabel.Location = new System.Drawing.Point(0, 1);
      this.demoLabel.Name = "demoLabel";
      this.demoLabel.Padding = new System.Windows.Forms.Padding(9);
      this.demoLabel.Size = new System.Drawing.Size(263, 348);
      this.demoLabel.TabIndex = 0;
      this.demoLabel.Text = resources.GetString("demoLabel.Text");
      // 
      // eventsListBox
      // 
      this.eventsListBox.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
      this.eventsListBox.FormattingEnabled = true;
      this.eventsListBox.Location = new System.Drawing.Point(3, 448);
      this.eventsListBox.Name = "eventsListBox";
      this.eventsListBox.Size = new System.Drawing.Size(260, 108);
      this.eventsListBox.TabIndex = 2;
      // 
      // menuStrip
      // 
      this.menuStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem,
            this.helpToolStripMenuItem});
      this.menuStrip.Location = new System.Drawing.Point(0, 0);
      this.menuStrip.Name = "menuStrip";
      this.menuStrip.Size = new System.Drawing.Size(784, 24);
      this.menuStrip.TabIndex = 0;
      // 
      // fileToolStripMenuItem
      // 
      this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.closeToolStripMenuItem});
      this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
      this.fileToolStripMenuItem.Size = new System.Drawing.Size(37, 20);
      this.fileToolStripMenuItem.Text = "&File";
      // 
      // closeToolStripMenuItem
      // 
      this.closeToolStripMenuItem.Name = "closeToolStripMenuItem";
      this.closeToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.W)));
      this.closeToolStripMenuItem.Size = new System.Drawing.Size(148, 22);
      this.closeToolStripMenuItem.Text = "&Close";
      this.closeToolStripMenuItem.Click += new System.EventHandler(this.closeToolStripMenuItem_Click);
      // 
      // helpToolStripMenuItem
      // 
      this.helpToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.aboutToolStripMenuItem});
      this.helpToolStripMenuItem.Name = "helpToolStripMenuItem";
      this.helpToolStripMenuItem.Size = new System.Drawing.Size(44, 20);
      this.helpToolStripMenuItem.Text = "&Help";
      // 
      // aboutToolStripMenuItem
      // 
      this.aboutToolStripMenuItem.Name = "aboutToolStripMenuItem";
      this.aboutToolStripMenuItem.Size = new System.Drawing.Size(116, 22);
      this.aboutToolStripMenuItem.Text = "&About...";
      this.aboutToolStripMenuItem.Click += new System.EventHandler(this.aboutToolStripMenuItem_Click);
      // 
      // statusStrip
      // 
      this.statusStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1,
            this.positionToolStripStatusLabel});
      this.statusStrip.Location = new System.Drawing.Point(0, 579);
      this.statusStrip.Name = "statusStrip";
      this.statusStrip.Size = new System.Drawing.Size(784, 22);
      this.statusStrip.TabIndex = 2;
      // 
      // toolStripStatusLabel1
      // 
      this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
      this.toolStripStatusLabel1.Size = new System.Drawing.Size(769, 17);
      this.toolStripStatusLabel1.Spring = true;
      // 
      // positionToolStripStatusLabel
      // 
      this.positionToolStripStatusLabel.Name = "positionToolStripStatusLabel";
      this.positionToolStripStatusLabel.Size = new System.Drawing.Size(0, 17);
      // 
      // PanStylesDemoForm
      // 
      this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
      this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
      this.ClientSize = new System.Drawing.Size(784, 601);
      this.Controls.Add(this.splitContainer);
      this.Controls.Add(this.menuStrip);
      this.Controls.Add(this.statusStrip);
      this.Name = "PanStylesDemoForm";
      this.Text = "Panning";
      this.splitContainer.Panel1.ResumeLayout(false);
      this.splitContainer.Panel2.ResumeLayout(false);
      this.splitContainer.ResumeLayout(false);
      this.menuStrip.ResumeLayout(false);
      this.menuStrip.PerformLayout();
      this.statusStrip.ResumeLayout(false);
      this.statusStrip.PerformLayout();
      this.ResumeLayout(false);
      this.PerformLayout();

    }

    #endregion

    private ImageBox imageBox;
    private System.Windows.Forms.Label demoLabel;
    private System.Windows.Forms.SplitContainer splitContainer;
    private System.Windows.Forms.MenuStrip menuStrip;
    private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
    private System.Windows.Forms.ToolStripMenuItem closeToolStripMenuItem;
    private System.Windows.Forms.ToolStripMenuItem helpToolStripMenuItem;
    private System.Windows.Forms.ToolStripMenuItem aboutToolStripMenuItem;
    private System.Windows.Forms.StatusStrip statusStrip;
    private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
    private System.Windows.Forms.ToolStripStatusLabel positionToolStripStatusLabel;
    private EventsListBox eventsListBox;
    private FilteredPropertyGrid propertyGrid;
  }
}
