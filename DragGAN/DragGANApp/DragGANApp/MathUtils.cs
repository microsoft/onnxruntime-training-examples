using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DragGANApp
{
    public static class MathUtils
    {
        public static T Clamp<T>(this T val, T min, T max) where T : IComparable<T>
        {
            if (val.CompareTo(min) < 0) return min;
            else if (val.CompareTo(max) > 0) return max;
            else return val;
        }

        public static float[] RotateVector2d(float x, float y, float degrees)
        {
            float[] result = new float[2];
            result[0] = x * (float)Math.Cos(degrees) - y * (float)Math.Sin(degrees);
            result[1] = x * (float)Math.Sin(degrees) + y * (float)Math.Cos(degrees);
            return result;
        }

        public static float Length(PointF p)
        {
            return (float)Math.Sqrt(p.X * p.X + p.Y * p.Y);
        }

        public static float Distance(PointF p1, PointF p2)
        {
            return (float)Math.Sqrt((p1.X - p2.X) * (p1.X - p2.X) + (p1.Y - p2.Y) * (p1.Y - p2.Y));
        }

    }

}
