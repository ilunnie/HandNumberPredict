using System;
using System.Collections.Generic;
using System.IO;
using Python.Runtime;

Runtime.PythonDLL = "python311.dll";
PythonEngine.Initialize();

dynamic tf = Py.Import("tensorflow");
dynamic np = Py.Import("numpy");
dynamic cv = Py.Import("cv2");

string model_path = "checkpoints/";

dynamic normal_model = tf.keras.models.load_model(model_path + "normal_model.keras");
dynamic binarized_model = tf.keras.models.load_model(model_path + "binarized_model.keras");
dynamic fourier_model = tf.keras.models.load_model(model_path + "fourier_model.keras");
dynamic binfourier_model = tf.keras.models.load_model(model_path + "bin&fourier_model.keras");

string folder = "./images/tests";
if (Directory.Exists(folder))
{
    string[] files = Directory.GetFiles(folder);
    List<PyObject> images = new List<PyObject>();
    foreach (string file in files)
    {
        dynamic i = Utils.ImRead(file);
        images.Add(Utils.Binarize(i));
    }
    PyObject result = cv.hconcat(images);

    cv.imshow("Result", result);
    cv.waitKey();
}


dynamic img = Utils.ImRead("images/tests/dedo1.jpeg");
dynamic data = np.expand_dims(img, axis: 0);

Console.WriteLine(normal_model.predict(data));
Console.WriteLine(binarized_model.predict(data));
Console.WriteLine(fourier_model.predict(data));
Console.WriteLine(binfourier_model.predict(data));

cv.imshow(img);
cv.waitKey();

Console.WriteLine(img.GetType());

PythonEngine.Shutdown();

