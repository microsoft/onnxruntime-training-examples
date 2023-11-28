using System;
using System.Collections.Generic;
using System.ComponentModel;

// Derived from https://www.codeproject.com/articles/13342/filtering-properties-in-a-propertygrid

namespace Cyotek.Windows.Forms.Demo
{
  /// <summary>
  /// This class overrides the standard PropertyGrid provided by Microsoft.
  /// It also allows to hide (or filter) the properties of the SelectedObject displayed by the PropertyGrid.
  /// </summary>
  internal class FilteredPropertyGrid : PropertyGrid
  {
    #region Private Fields

    private AttributeCollection _browsableAttributes;

    private string[] _browsableProperties;

    private AttributeCollection _hiddenAttributes;

    private string[] _hiddenProperties;

    private List<PropertyDescriptor> _propertyDescriptors;

    private ObjectWrapper _wrapper;

    #endregion Private Fields

    #region Public Constructors

    /// <summary>Public constructor.</summary>
    public FilteredPropertyGrid()
    {
      _propertyDescriptors = new List<PropertyDescriptor>();
    }

    #endregion Public Constructors

    #region Public Properties

    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public new AttributeCollection BrowsableAttributes
    {
      get { return _browsableAttributes; }
      set
      {
        if (_browsableAttributes != value)
        {
          _hiddenAttributes = null;
          _browsableAttributes = value;
          this.RefreshProperties();
        }
      }
    }

    /// <summary>Get or set the properties to show.</summary>
    /// <exception cref="ArgumentException">if one or several properties don't exist.</exception>
    [DefaultValue(typeof(string[]), null)]
    public string[] BrowsableProperties
    {
      get { return _browsableProperties; }
      set
      {
        if (value != _browsableProperties)
        {
          _browsableProperties = value;
          //m_HiddenProperties = null;
          this.RefreshProperties();
        }
      }
    }

    /// <summary>Get or set the categories to hide.</summary>
    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public AttributeCollection HiddenAttributes
    {
      get { return _hiddenAttributes; }
      set
      {
        if (value != _hiddenAttributes)
        {
          _hiddenAttributes = value;
          _browsableAttributes = null;
          this.RefreshProperties();
        }
      }
    }

    /// <summary>Get or set the properties to hide.</summary>
    [DefaultValue(typeof(string[]), null)]
    public string[] HiddenProperties
    {
      get { return _hiddenProperties; }
      set
      {
        if (value != _hiddenProperties)
        {
          //m_BrowsableProperties = null;
          _hiddenProperties = value;
          this.RefreshProperties();
        }
      }
    }

    /// <summary>Overwrite the PropertyGrid.SelectedObject property.</summary>
    /// <remarks>The object passed to the base PropertyGrid is the wrapper.</remarks>
    public new object SelectedObject
    {
      get { return _wrapper != null ? ((ObjectWrapper)base.SelectedObject).SelectedObject : null; }
      set
      {
        // Set the new object to the wrapper and create one if necessary.
        if (_wrapper == null)
        {
          _wrapper = new ObjectWrapper(value);
          this.RefreshProperties();
        }
        else if (_wrapper.SelectedObject != value)
        {
          bool needrefresh = value.GetType() != _wrapper.SelectedObject.GetType();
          _wrapper.SelectedObject = value;
          if (needrefresh) this.RefreshProperties();
        }

        // Set the list of properties to the wrapper.
        _wrapper.PropertyDescriptors = _propertyDescriptors;

        // Link the wrapper to the parent PropertyGrid.
        base.SelectedObject = _wrapper;
      }
    }

    #endregion Public Properties

    #region Private Methods

    /// <summary>Allows to hide a set of properties to the parent PropertyGrid.</summary>
    /// <param name="propertyname">A set of attributes that filter the original collection of properties.</param>
    /// <remarks>For better performance, include the BrowsableAttribute with true value.</remarks>
    private void HideAttribute(Attribute attribute)
    {
      PropertyDescriptorCollection filteredoriginalpropertydescriptors;

      filteredoriginalpropertydescriptors = TypeDescriptor.GetProperties(_wrapper.SelectedObject, new Attribute[] { attribute });

      if (filteredoriginalpropertydescriptors == null || filteredoriginalpropertydescriptors.Count == 0)
      {
        throw new ArgumentException("Attribute not found", attribute.ToString());
      }

      foreach (PropertyDescriptor propertydescriptor in filteredoriginalpropertydescriptors)
      {
        this.HideProperty(propertydescriptor);
      }
    }

    /// <summary>Allows to hide a property to the parent PropertyGrid.</summary>
    /// <param name="propertyname">The name of the property to be hidden.</param>
    private void HideProperty(PropertyDescriptor property)
    {
      if (_propertyDescriptors.Contains(property))
      {
        _propertyDescriptors.Remove(property);
      }
    }

    /// <summary>Build the list of the properties to be displayed in the PropertyGrid, following the filters defined the Browsable and Hidden properties.</summary>
    private void RefreshProperties()
    {
      if (_wrapper != null)
      {
        PropertyDescriptorCollection allproperties;

        // Clear the list of properties to be displayed.
        _propertyDescriptors.Clear();

        // Check whether the list is filtered
        if (_browsableAttributes != null && _browsableAttributes.Count > 0)
        {
          // Add to the list the attributes that need to be displayed.
          foreach (Attribute attribute in _browsableAttributes)
          {
            this.ShowAttribute(attribute);
          }
        }
        else
        {
          // Fill the collection with all the properties.
          //   PropertyDescriptorCollection originalpropertydescriptors = TypeDescriptor.GetProperties(m_Wrapper.SelectedObject);
          //  foreach (PropertyDescriptor propertydescriptor in originalpropertydescriptors) m_PropertyDescriptors.Add(propertydescriptor);
          // Remove from the list the attributes that mustn't be displayed.
          if (_hiddenAttributes != null)
          {
            foreach (Attribute attribute in _hiddenAttributes)
            {
              this.HideAttribute(attribute);
            }
          }
        }

        // Get all the properties of the SelectedObject
        allproperties = TypeDescriptor.GetProperties(_wrapper.SelectedObject);

        // Hide if necessary, some properties
        if (_hiddenProperties != null && _hiddenProperties.Length > 0)
        {
          // Remove from the list the properties that mustn't be displayed.
          foreach (string propertyname in _hiddenProperties)
          {
            // Remove from the list the property
            this.HideProperty(allproperties[propertyname]);
          }
        }
        // Display if necessary, some properties
        if (_browsableProperties != null && _browsableProperties.Length > 0)
        {
          foreach (string propertyname in _browsableProperties)
          {
            this.ShowProperty(allproperties[propertyname]);
          }
        }
      }
    }

    /// <summary>Add all the properties that match an attribute to the list of properties to be displayed in the PropertyGrid.</summary>
    /// <param name="property">The attribute to be added.</param>
    private void ShowAttribute(Attribute attribute)
    {
      PropertyDescriptorCollection filteredoriginalpropertydescriptors;

      filteredoriginalpropertydescriptors = TypeDescriptor.GetProperties(_wrapper.SelectedObject, new Attribute[] { attribute });

      if (filteredoriginalpropertydescriptors == null || filteredoriginalpropertydescriptors.Count == 0)
      {
        throw new ArgumentException("Attribute not found", attribute.ToString());
      }

      foreach (PropertyDescriptor propertydescriptor in filteredoriginalpropertydescriptors)
      {
        this.ShowProperty(propertydescriptor);
      }
    }

    /// <summary>Add a property to the list of properties to be displayed in the PropertyGrid.</summary>
    /// <param name="property">The property to be added.</param>
    private void ShowProperty(PropertyDescriptor property)
    {
      if (!_propertyDescriptors.Contains(property))
      {
        _propertyDescriptors.Add(property);
      }
    }

    #endregion Private Methods

    #region Private Classes

    /// <summary>This class is a wrapper. It contains the object the propertyGrid has to display.</summary>
    private sealed class ObjectWrapper : ICustomTypeDescriptor
    {
      #region Private Fields

      private List<PropertyDescriptor> _propertyDescriptors;

      private object _selectedObject;

      #endregion Private Fields

      #region Internal Constructors

      /// <summary>Simple constructor.</summary>
      /// <param name="obj">A reference to the selected object that will linked to the parent PropertyGrid.</param>
      internal ObjectWrapper(object obj)
      {
        _propertyDescriptors = new List<PropertyDescriptor>();
        _selectedObject = obj;
      }

      #endregion Internal Constructors

      #region Public Properties

      /// <summary>Get or set a reference to the collection of properties to show in the parent PropertyGrid.</summary>
      public List<PropertyDescriptor> PropertyDescriptors
      {
        get { return _propertyDescriptors; }
        set { _propertyDescriptors = value; }
      }

      /// <summary>Get or set a reference to the selected objet that will linked to the parent PropertyGrid.</summary>
      public object SelectedObject
      {
        get { return _selectedObject; }
        set
        {
          if (_selectedObject != value)
          {
            _selectedObject = value;
          }
        }
      }

      #endregion Public Properties

      #region Public Methods

      /// <summary>GetAttributes.</summary>
      /// <returns>AttributeCollection</returns>
      public AttributeCollection GetAttributes()
      {
        return TypeDescriptor.GetAttributes(_selectedObject, true);
      }

      /// <summary>Get Class Name.</summary>
      /// <returns>String</returns>
      public string GetClassName()
      {
        return TypeDescriptor.GetClassName(_selectedObject, true);
      }

      /// <summary>GetComponentName.</summary>
      /// <returns>String</returns>
      public string GetComponentName()
      {
        return TypeDescriptor.GetComponentName(_selectedObject, true);
      }

      /// <summary>GetConverter.</summary>
      /// <returns>TypeConverter</returns>
      public TypeConverter GetConverter()
      {
        return TypeDescriptor.GetConverter(_selectedObject, true);
      }

      /// <summary>GetDefaultEvent.</summary>
      /// <returns>EventDescriptor</returns>
      public EventDescriptor GetDefaultEvent()
      {
        return TypeDescriptor.GetDefaultEvent(_selectedObject, true);
      }

      /// <summary>GetDefaultProperty.</summary>
      /// <returns>PropertyDescriptor</returns>
      public PropertyDescriptor GetDefaultProperty()
      {
        return TypeDescriptor.GetDefaultProperty(_selectedObject, true);
      }

      /// <summary>GetEditor.</summary>
      /// <param name="editorBaseType">editorBaseType</param>
      /// <returns>object</returns>
      public object GetEditor(Type editorBaseType)
      {
        return TypeDescriptor.GetEditor(this, editorBaseType, true);
      }

      public EventDescriptorCollection GetEvents(Attribute[] attributes)
      {
        return TypeDescriptor.GetEvents(_selectedObject, attributes, true);
      }

      public EventDescriptorCollection GetEvents()
      {
        return TypeDescriptor.GetEvents(_selectedObject, true);
      }

      public PropertyDescriptorCollection GetProperties(Attribute[] attributes)
      {
        return this.GetProperties();
      }

      public PropertyDescriptorCollection GetProperties()
      {
        return new PropertyDescriptorCollection(_propertyDescriptors.ToArray(), true);
      }

      public object GetPropertyOwner(PropertyDescriptor pd)
      {
        return _selectedObject;
      }

      #endregion Public Methods
    }

    #endregion Private Classes
  }
}
