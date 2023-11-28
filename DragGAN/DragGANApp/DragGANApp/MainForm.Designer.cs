namespace DragGANApp
{
    partial class MainForm
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
            this.menuStripMain = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.runToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.undoonelyOnceToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.redoToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.stopOptimizationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.exitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.panelTop = new System.Windows.Forms.Panel();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.styleGANSamplesContainer1 = new DragGANApp.Tiles.StyleGANSamplesContainer();
            this.splitContainer2 = new System.Windows.Forms.SplitContainer();
            this.imageBoxEdit = new Cyotek.Windows.Forms.ImageBox();
            this.panel1 = new System.Windows.Forms.Panel();
            this.flowLayoutPanel1 = new System.Windows.Forms.FlowLayoutPanel();
            this.buttonOptimize = new System.Windows.Forms.Button();
            this.buttonStop = new System.Windows.Forms.Button();
            this.buttonUndo = new System.Windows.Forms.Button();
            this.buttonRedo = new System.Windows.Forms.Button();
            this.imageBoxView = new Cyotek.Windows.Forms.ImageBox();
            this.menuStripMain.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).BeginInit();
            this.splitContainer2.Panel1.SuspendLayout();
            this.splitContainer2.Panel2.SuspendLayout();
            this.splitContainer2.SuspendLayout();
            this.panel1.SuspendLayout();
            this.flowLayoutPanel1.SuspendLayout();
            this.SuspendLayout();
            // 
            // menuStripMain
            // 
            this.menuStripMain.GripMargin = new System.Windows.Forms.Padding(2, 2, 0, 2);
            this.menuStripMain.ImageScalingSize = new System.Drawing.Size(24, 24);
            this.menuStripMain.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem});
            this.menuStripMain.Location = new System.Drawing.Point(5, 5);
            this.menuStripMain.Name = "menuStripMain";
            this.menuStripMain.Padding = new System.Windows.Forms.Padding(12, 4, 0, 4);
            this.menuStripMain.Size = new System.Drawing.Size(1980, 37);
            this.menuStripMain.TabIndex = 0;
            this.menuStripMain.Text = "menuStrip1";
            this.menuStripMain.Visible = false;
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.runToolStripMenuItem,
            this.undoonelyOnceToolStripMenuItem,
            this.redoToolStripMenuItem,
            this.stopOptimizationToolStripMenuItem,
            this.exitToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(54, 29);
            this.fileToolStripMenuItem.Text = "File";
            // 
            // runToolStripMenuItem
            // 
            this.runToolStripMenuItem.Name = "runToolStripMenuItem";
            this.runToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.R)));
            this.runToolStripMenuItem.Size = new System.Drawing.Size(324, 34);
            this.runToolStripMenuItem.Text = "Run";
            this.runToolStripMenuItem.Click += new System.EventHandler(this.runToolStripMenuItem_Click);
            // 
            // undoonelyOnceToolStripMenuItem
            // 
            this.undoonelyOnceToolStripMenuItem.Name = "undoonelyOnceToolStripMenuItem";
            this.undoonelyOnceToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Z)));
            this.undoonelyOnceToolStripMenuItem.Size = new System.Drawing.Size(324, 34);
            this.undoonelyOnceToolStripMenuItem.Text = "Undo";
            this.undoonelyOnceToolStripMenuItem.Click += new System.EventHandler(this.undoonelyOnceToolStripMenuItem_Click);
            // 
            // redoToolStripMenuItem
            // 
            this.redoToolStripMenuItem.Name = "redoToolStripMenuItem";
            this.redoToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Y)));
            this.redoToolStripMenuItem.Size = new System.Drawing.Size(324, 34);
            this.redoToolStripMenuItem.Text = "Redo";
            this.redoToolStripMenuItem.Click += new System.EventHandler(this.redoToolStripMenuItem_Click);
            // 
            // stopOptimizationToolStripMenuItem
            // 
            this.stopOptimizationToolStripMenuItem.Name = "stopOptimizationToolStripMenuItem";
            this.stopOptimizationToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Q)));
            this.stopOptimizationToolStripMenuItem.Size = new System.Drawing.Size(324, 34);
            this.stopOptimizationToolStripMenuItem.Text = "Stop Optimization";
            this.stopOptimizationToolStripMenuItem.Click += new System.EventHandler(this.stopOptimizationToolStripMenuItem_Click);
            // 
            // exitToolStripMenuItem
            // 
            this.exitToolStripMenuItem.Name = "exitToolStripMenuItem";
            this.exitToolStripMenuItem.Size = new System.Drawing.Size(324, 34);
            this.exitToolStripMenuItem.Text = "Exit";
            this.exitToolStripMenuItem.Click += new System.EventHandler(this.exitToolStripMenuItem_Click);
            // 
            // panelTop
            // 
            this.panelTop.Dock = System.Windows.Forms.DockStyle.Top;
            this.panelTop.Location = new System.Drawing.Point(5, 5);
            this.panelTop.Margin = new System.Windows.Forms.Padding(5, 6, 5, 6);
            this.panelTop.Name = "panelTop";
            this.panelTop.Size = new System.Drawing.Size(1980, 78);
            this.panelTop.TabIndex = 2;
            this.panelTop.Visible = false;
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(5, 83);
            this.splitContainer1.Margin = new System.Windows.Forms.Padding(4);
            this.splitContainer1.Name = "splitContainer1";
            this.splitContainer1.Orientation = System.Windows.Forms.Orientation.Horizontal;
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.styleGANSamplesContainer1);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.splitContainer2);
            this.splitContainer1.Size = new System.Drawing.Size(1980, 890);
            this.splitContainer1.SplitterDistance = 162;
            this.splitContainer1.TabIndex = 5;
            // 
            // styleGANSamplesContainer1
            // 
            this.styleGANSamplesContainer1.BatchSize = 10;
            this.styleGANSamplesContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.styleGANSamplesContainer1.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.styleGANSamplesContainer1.GroupHeight = 0;
            this.styleGANSamplesContainer1.Location = new System.Drawing.Point(0, 0);
            this.styleGANSamplesContainer1.Margin = new System.Windows.Forms.Padding(5, 4, 5, 4);
            this.styleGANSamplesContainer1.Name = "styleGANSamplesContainer1";
            this.styleGANSamplesContainer1.Size = new System.Drawing.Size(1980, 162);
            this.styleGANSamplesContainer1.TabIndex = 1;
            // 
            // splitContainer2
            // 
            this.splitContainer2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer2.Location = new System.Drawing.Point(0, 0);
            this.splitContainer2.Name = "splitContainer2";
            // 
            // splitContainer2.Panel1
            // 
            this.splitContainer2.Panel1.Controls.Add(this.imageBoxEdit);
            // 
            // splitContainer2.Panel2
            // 
            this.splitContainer2.Panel2.Controls.Add(this.panel1);
            this.splitContainer2.Panel2.Controls.Add(this.imageBoxView);
            this.splitContainer2.Size = new System.Drawing.Size(1980, 724);
            this.splitContainer2.SplitterDistance = 1387;
            this.splitContainer2.TabIndex = 5;
            // 
            // imageBoxEdit
            // 
            this.imageBoxEdit.Dock = System.Windows.Forms.DockStyle.Fill;
            this.imageBoxEdit.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.imageBoxEdit.Location = new System.Drawing.Point(0, 0);
            this.imageBoxEdit.Margin = new System.Windows.Forms.Padding(11);
            this.imageBoxEdit.Name = "imageBoxEdit";
            this.imageBoxEdit.Size = new System.Drawing.Size(1387, 724);
            this.imageBoxEdit.TabIndex = 4;
            this.imageBoxEdit.TextAlign = System.Drawing.ContentAlignment.TopLeft;
            this.imageBoxEdit.TextPadding = new System.Windows.Forms.Padding(5);
            this.imageBoxEdit.MouseClick += new System.Windows.Forms.MouseEventHandler(this.imageBoxEdit_MouseClick);
            this.imageBoxEdit.MouseDown += new System.Windows.Forms.MouseEventHandler(this.imageBoxEdit_MouseDown);
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.flowLayoutPanel1);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.panel1.Location = new System.Drawing.Point(0, 680);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(589, 44);
            this.panel1.TabIndex = 6;
            // 
            // flowLayoutPanel1
            // 
            this.flowLayoutPanel1.Controls.Add(this.buttonOptimize);
            this.flowLayoutPanel1.Controls.Add(this.buttonStop);
            this.flowLayoutPanel1.Controls.Add(this.buttonUndo);
            this.flowLayoutPanel1.Controls.Add(this.buttonRedo);
            this.flowLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.flowLayoutPanel1.Location = new System.Drawing.Point(0, 0);
            this.flowLayoutPanel1.Name = "flowLayoutPanel1";
            this.flowLayoutPanel1.Size = new System.Drawing.Size(589, 44);
            this.flowLayoutPanel1.TabIndex = 0;
            // 
            // buttonOptimize
            // 
            this.buttonOptimize.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.buttonOptimize.Location = new System.Drawing.Point(3, 3);
            this.buttonOptimize.Name = "buttonOptimize";
            this.buttonOptimize.Size = new System.Drawing.Size(120, 40);
            this.buttonOptimize.TabIndex = 0;
            this.buttonOptimize.Text = "Optimize";
            this.buttonOptimize.UseVisualStyleBackColor = true;
            this.buttonOptimize.Click += new System.EventHandler(this.buttonOptimize_Click);
            // 
            // buttonStop
            // 
            this.buttonStop.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.buttonStop.Location = new System.Drawing.Point(129, 3);
            this.buttonStop.Name = "buttonStop";
            this.buttonStop.Size = new System.Drawing.Size(120, 40);
            this.buttonStop.TabIndex = 1;
            this.buttonStop.Text = "Stop";
            this.buttonStop.UseVisualStyleBackColor = true;
            this.buttonStop.Click += new System.EventHandler(this.buttonStop_Click);
            // 
            // buttonUndo
            // 
            this.buttonUndo.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.buttonUndo.Location = new System.Drawing.Point(255, 3);
            this.buttonUndo.Name = "buttonUndo";
            this.buttonUndo.Size = new System.Drawing.Size(120, 40);
            this.buttonUndo.TabIndex = 2;
            this.buttonUndo.Text = "Undo";
            this.buttonUndo.UseVisualStyleBackColor = true;
            this.buttonUndo.Click += new System.EventHandler(this.buttonUndo_Click);
            // 
            // buttonRedo
            // 
            this.buttonRedo.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.buttonRedo.Location = new System.Drawing.Point(381, 3);
            this.buttonRedo.Name = "buttonRedo";
            this.buttonRedo.Size = new System.Drawing.Size(120, 40);
            this.buttonRedo.TabIndex = 3;
            this.buttonRedo.Text = "Redo";
            this.buttonRedo.UseVisualStyleBackColor = true;
            this.buttonRedo.Click += new System.EventHandler(this.buttonRedo_Click);
            // 
            // imageBoxView
            // 
            this.imageBoxView.Dock = System.Windows.Forms.DockStyle.Fill;
            this.imageBoxView.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.imageBoxView.Location = new System.Drawing.Point(0, 0);
            this.imageBoxView.Margin = new System.Windows.Forms.Padding(11);
            this.imageBoxView.Name = "imageBoxView";
            this.imageBoxView.Size = new System.Drawing.Size(589, 724);
            this.imageBoxView.TabIndex = 5;
            this.imageBoxView.Text = "View";
            this.imageBoxView.TextAlign = System.Drawing.ContentAlignment.TopLeft;
            this.imageBoxView.TextPadding = new System.Windows.Forms.Padding(5);
            this.imageBoxView.SizeChanged += new System.EventHandler(this.imageBoxView_SizeChanged);
            this.imageBoxView.MouseDown += new System.Windows.Forms.MouseEventHandler(this.imageBoxView_MouseDown);
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(12F, 25F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(255)))), ((int)(((byte)(222)))));
            this.ClientSize = new System.Drawing.Size(1990, 978);
            this.Controls.Add(this.splitContainer1);
            this.Controls.Add(this.panelTop);
            this.Controls.Add(this.menuStripMain);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.MainMenuStrip = this.menuStripMain;
            this.Margin = new System.Windows.Forms.Padding(4);
            this.Name = "MainForm";
            this.Padding = new System.Windows.Forms.Padding(5);
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "DragGAN App";
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.MainForm_FormClosed);
            this.Load += new System.EventHandler(this.MainForm_Load);
            this.Shown += new System.EventHandler(this.MainForm_Shown);
            this.menuStripMain.ResumeLayout(false);
            this.menuStripMain.PerformLayout();
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.splitContainer2.Panel1.ResumeLayout(false);
            this.splitContainer2.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).EndInit();
            this.splitContainer2.ResumeLayout(false);
            this.panel1.ResumeLayout(false);
            this.flowLayoutPanel1.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip menuStripMain;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.Panel panelTop;
        private Cyotek.Windows.Forms.ImageBox imageBoxEdit;
        private System.Windows.Forms.ToolStripMenuItem runToolStripMenuItem;
        private System.Windows.Forms.SplitContainer splitContainer1;
        private Tiles.StyleGANSamplesContainer styleGANSamplesContainer1;
        private System.Windows.Forms.SplitContainer splitContainer2;
        private Cyotek.Windows.Forms.ImageBox imageBoxView;
        private System.Windows.Forms.ToolStripMenuItem undoonelyOnceToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem exitToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem stopOptimizationToolStripMenuItem;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.FlowLayoutPanel flowLayoutPanel1;
        private System.Windows.Forms.Button buttonOptimize;
        private System.Windows.Forms.Button buttonStop;
        private System.Windows.Forms.Button buttonUndo;
        private System.Windows.Forms.ToolStripMenuItem redoToolStripMenuItem;
        private System.Windows.Forms.Button buttonRedo;
    }
}

