namespace Cyotek.Windows.Forms
{
  /// <summary>
  /// Specifies the way the mouse wheel is handled for the <see cref="ImageBox"/> control.
  /// </summary>
  public enum ImageBoxMouseWheelMode
  {
    /// <summary>
    /// Mouse wheel not handled
    /// </summary>
    None,

    /// <summary>
    /// Mouse wheel zooms
    /// </summary>
    Zoom,

    /// <summary>
    /// Mouse wheel scrolls vertically, Shift + mouse wheel scrolls horizontally and Ctrl + mouse wheel zooms
    /// </summary>
    ScrollAndZoom = 3
  }
}
