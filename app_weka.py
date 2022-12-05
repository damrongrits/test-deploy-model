#heroku buildpacks:set heroku/python
#heroku buildpacks:add --index 1 heroku/jvm

import weka.core.serialization as serialization
from weka.classifiers import Classifier
from weka.core.converters import Loader
from weka.core.dataset import Attribute, Instance, Instances
import weka.core.jvm as jvm

from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index_weka.html')

@app.route('/predict', methods = ['POST'])
def getPredict():
    x1 = request.form['x1']
    x2 = request.form['x2']
    x3 = request.form['x3']
    x4 = request.form['x4']
    x5 = request.form['x5']
    x6 = request.form['x6']

    #os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-11.0.10.jdk/Contents/Home"
    jvm.start()
    objects = serialization.read_all("PMj48.model")
    classifier = Classifier(jobject=objects[0])
    loader = Loader (classname = "weka.core.converters.ArffLoader")
    # create attributes
    type_att = Attribute.create_nominal("Type", ["M", "L", "H"])
    num1_att = Attribute.create_numeric("Air temperature [K]")
    num2_att = Attribute.create_numeric("Process temperature [K]")
    num3_att = Attribute.create_numeric("Rotational speed [rpm]")
    num4_att = Attribute.create_numeric("Torque [Nm]")
    num5_att = Attribute.create_numeric("Tool wear [min]")
    MF_att = Attribute.create_nominal("MF", ["Normal", "Failure"])
    # create dataset
    dataset = Instances.create_instances("helloworld", [type_att, num1_att, num2_att, num3_att, num4_att, num5_att, MF_att], 0)
    dataset.class_is_last()
    # add rows
    values = [x1,x2,x3,x4,x5,x6,0]
    inst = Instance.create_instance(values)
    dataset.add_instance(inst)

    predicted = classifier.classify_instance(dataset[0])

    if (predicted==0.0):
        output="Normal"
    else:
        output="Failure"
    jvm.stop()
    return render_template('index_weka.html', prediction_text = f'Predicted (MF): {output}')

if __name__ == '__main__':
    app.run(debug = True)