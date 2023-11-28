using System;
using System.Windows.Forms;

namespace Cyotek.Windows.Forms
{
  /// <summary>
  /// Specifies constants that define which mouse buttons can be used to pan an <see cref="ImageBox"/> control.
  /// </summary>
  [Flags]
  public enum ImageBoxPanMode
  {
    /// <summary>
    /// No mouse buttons can be used to pan the control.
    /// </summary>
    None = 0,

    /// <summary>
    /// The left mouse button can be used to pan the control.
    /// </summary>
    Left = MouseButtons.Left,

    /// <summary>
    /// The middle mouse button can be used to pan the control.
    /// </summary>
    Middle = MouseButtons.Middle,

    /// <summary>
    /// Both the left and left mouse buttons can be used to pan the control.
    /// </summary>
    Both = Left | Middle
  }
}
