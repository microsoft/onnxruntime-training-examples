namespace DragGANApp.Tiles
{
    partial class StyleGANSamplesContainer
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(StyleGANSamplesContainer));
            this.panelControls = new System.Windows.Forms.Panel();
            this.flowLayoutPanelControls = new System.Windows.Forms.FlowLayoutPanel();
            this.buttonNext = new System.Windows.Forms.Button();
            this.buttonPrevious = new System.Windows.Forms.Button();
            this.buttonGoto = new System.Windows.Forms.Button();
            this.numericUpDownSeed = new System.Windows.Forms.NumericUpDown();
            this.flowLayoutPanelItems = new System.Windows.Forms.FlowLayoutPanel();
            this.imageList1 = new System.Windows.Forms.ImageList(this.components);
            this.panelControls.SuspendLayout();
            this.flowLayoutPanelControls.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownSeed)).BeginInit();
            this.SuspendLayout();
            // 
            // panelControls
            // 
            this.panelControls.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(255)))), ((int)(((byte)(192)))));
            this.panelControls.Controls.Add(this.flowLayoutPanelControls);
            this.panelControls.Dock = System.Windows.Forms.DockStyle.Right;
            this.panelControls.Location = new System.Drawing.Point(1240, 0);
            this.panelControls.Margin = new System.Windows.Forms.Padding(4);
            this.panelControls.Name = "panelControls";
            this.panelControls.Size = new System.Drawing.Size(187, 352);
            this.panelControls.TabIndex = 0;
            // 
            // flowLayoutPanelControls
            // 
            this.flowLayoutPanelControls.BackColor = System.Drawing.SystemColors.Control;
            this.flowLayoutPanelControls.Controls.Add(this.buttonNext);
            this.flowLayoutPanelControls.Controls.Add(this.buttonPrevious);
            this.flowLayoutPanelControls.Controls.Add(this.buttonGoto);
            this.flowLayoutPanelControls.Controls.Add(this.numericUpDownSeed);
            this.flowLayoutPanelControls.Dock = System.Windows.Forms.DockStyle.Fill;
            this.flowLayoutPanelControls.FlowDirection = System.Windows.Forms.FlowDirection.TopDown;
            this.flowLayoutPanelControls.Location = new System.Drawing.Point(0, 0);
            this.flowLayoutPanelControls.Margin = new System.Windows.Forms.Padding(4);
            this.flowLayoutPanelControls.Name = "flowLayoutPanelControls";
            this.flowLayoutPanelControls.Size = new System.Drawing.Size(187, 352);
            this.flowLayoutPanelControls.TabIndex = 0;
            // 
            // buttonNext
            // 
            this.buttonNext.BackgroundImageLayout = System.Windows.Forms.ImageLayout.None;
            this.buttonNext.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.buttonNext.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonNext.ImageIndex = 1;
            this.buttonNext.Location = new System.Drawing.Point(4, 4);
            this.buttonNext.Margin = new System.Windows.Forms.Padding(4);
            this.buttonNext.Name = "buttonNext";
            this.buttonNext.Size = new System.Drawing.Size(160, 40);
            this.buttonNext.TabIndex = 0;
            this.buttonNext.Text = "Forward";
            this.buttonNext.UseVisualStyleBackColor = true;
            this.buttonNext.Click += new System.EventHandler(this.buttonNext_Click);
            // 
            // buttonPrevious
            // 
            this.buttonPrevious.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.buttonPrevious.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonPrevious.ImageIndex = 0;
            this.buttonPrevious.Location = new System.Drawing.Point(4, 52);
            this.buttonPrevious.Margin = new System.Windows.Forms.Padding(4);
            this.buttonPrevious.Name = "buttonPrevious";
            this.buttonPrevious.Size = new System.Drawing.Size(160, 40);
            this.buttonPrevious.TabIndex = 1;
            this.buttonPrevious.Text = "Back";
            this.buttonPrevious.UseVisualStyleBackColor = true;
            this.buttonPrevious.Click += new System.EventHandler(this.buttonPrevious_Click);
            // 
            // buttonGoto
            // 
            this.buttonGoto.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.buttonGoto.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonGoto.Location = new System.Drawing.Point(4, 100);
            this.buttonGoto.Margin = new System.Windows.Forms.Padding(4);
            this.buttonGoto.Name = "buttonGoto";
            this.buttonGoto.Size = new System.Drawing.Size(160, 40);
            this.buttonGoto.TabIndex = 4;
            this.buttonGoto.Text = "Goto";
            this.buttonGoto.UseVisualStyleBackColor = true;
            this.buttonGoto.Click += new System.EventHandler(this.buttonGoto_Click);
            // 
            // numericUpDownSeed
            // 
            this.numericUpDownSeed.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.numericUpDownSeed.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownSeed.Increment = new decimal(new int[] {
            10,
            0,
            0,
            0});
            this.numericUpDownSeed.Location = new System.Drawing.Point(4, 148);
            this.numericUpDownSeed.Margin = new System.Windows.Forms.Padding(4);
            this.numericUpDownSeed.Maximum = new decimal(new int[] {
            30000,
            0,
            0,
            0});
            this.numericUpDownSeed.Name = "numericUpDownSeed";
            this.numericUpDownSeed.Size = new System.Drawing.Size(160, 30);
            this.numericUpDownSeed.TabIndex = 3;
            // 
            // flowLayoutPanelItems
            // 
            this.flowLayoutPanelItems.AutoScroll = true;
            this.flowLayoutPanelItems.BackColor = System.Drawing.SystemColors.Control;
            this.flowLayoutPanelItems.Dock = System.Windows.Forms.DockStyle.Fill;
            this.flowLayoutPanelItems.Location = new System.Drawing.Point(0, 0);
            this.flowLayoutPanelItems.Margin = new System.Windows.Forms.Padding(4);
            this.flowLayoutPanelItems.Name = "flowLayoutPanelItems";
            this.flowLayoutPanelItems.Size = new System.Drawing.Size(1240, 352);
            this.flowLayoutPanelItems.TabIndex = 1;
            // 
            // imageList1
            // 
            this.imageList1.ImageStream = ((System.Windows.Forms.ImageListStreamer)(resources.GetObject("imageList1.ImageStream")));
            this.imageList1.TransparentColor = System.Drawing.Color.Transparent;
            this.imageList1.Images.SetKeyName(0, "arrow-left.png");
            this.imageList1.Images.SetKeyName(1, "arrow-right.png");
            this.imageList1.Images.SetKeyName(2, "down-arrow.png");
            // 
            // StyleGANSamplesContainer
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(12F, 25F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.flowLayoutPanelItems);
            this.Controls.Add(this.panelControls);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Margin = new System.Windows.Forms.Padding(4);
            this.Name = "StyleGANSamplesContainer";
            this.Size = new System.Drawing.Size(1427, 352);
            this.panelControls.ResumeLayout(false);
            this.flowLayoutPanelControls.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownSeed)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel panelControls;
        private System.Windows.Forms.FlowLayoutPanel flowLayoutPanelItems;
        private System.Windows.Forms.FlowLayoutPanel flowLayoutPanelControls;
        private System.Windows.Forms.Button buttonNext;
        private System.Windows.Forms.Button buttonPrevious;
        private System.Windows.Forms.NumericUpDown numericUpDownSeed;
        private System.Windows.Forms.Button buttonGoto;
        private System.Windows.Forms.ImageList imageList1;
    }
}
