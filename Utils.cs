using System;
using Python.Runtime;

public static class Utils
{
    public static dynamic np { get; private set; } = Py.Import("numpy");
    public static dynamic cv { get; private set; } = Py.Import("cv2");

    public static PyObject ImRead(string path, int width = 224, int height = 224)
    {
        PyObject img = cv.imread(path);
        img = img.Resize(width, height);

        return img;
    }

    public static PyObject Resize(this PyObject img, int width = 244, int height = 224)
    {
        dynamic size = new PyTuple(new PyObject[] { new PyInt(width), new PyInt(height) });
        return cv.resize(img, size);
    }

    public static PyObject Binarize(this PyObject img)
    {
        img = Filter(img,
            (byte b, byte g, byte r)
                =>  !(r > b &&
                    r > g &&
                    b > 40 &&
                    g > 40 &&
                    r > 40));
        // img = cv.cvtColor(img, cv.COLOR_BGR2GRAY);

        // img = cv.threshold(
        //     img, 100, 255, cv.THRESH_BINARY
        // )[1];

        // img = cv.dilate(img, np.ones((5, 5)));

        return img;
    }

    public static PyObject Filter(this PyObject img, Func<byte, byte, byte, bool> filter, byte grayscale = 0)
    {
        for (int j = 0; j < img.Length(); j++)
        {
            dynamic row = img[j];
            for (int i = 0; i < row.Length(); i++)
            {
                byte b = (byte)row[i][0];
                byte g = (byte)row[i][1];
                byte r = (byte)row[i][2];

                dynamic gray = new PyInt(grayscale);
                if (filter(b, g, r))
                    row[i] = new PyTuple(new PyObject[] {gray, gray, gray});
            }
        }
        return img;
    }
}