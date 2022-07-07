from django.shortcuts import render
from django.core.files.storage import default_storage

import os
import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np

from pyAudioAnalysis import audioTrainTest as aT

import requests

from COVIDvsML.models import Alerts

def home(request):
    return render(request,'index.html')

def about_covid(request):
    return render(request,'aboutcovid.html')

def contact_tracing(request):
    if request.method=="POST":
        place = request.POST['placeName']
        alerts = Alerts.objects.filter(venue__contains=place).order_by('-time_date')
    else:
        alerts = Alerts.objects.order_by('-time_date')

    data = {
        "alert": alerts
    }
    return render(request,'contact_tracing_alerts.html', data)

def cough_sound_pred(request):
    if request.method=="POST":
        f = request.FILES['coughSound']
        response = {}
        file_name = default_storage.save(f.name, f)
        media_dir = os.path.dirname(os.path.dirname(__file__))+'/media'
        audio_file = os.path.join(media_dir,file_name)
        model=os.path.dirname(__file__)+'/svm_model'
        pred = aT.file_classification(audio_file, model ,"svm")
        print(pred)
        response['pred']=pred[0]
        return render(request,'cough_sound_detection.html',response)
    else:
        return render(request,'cough_sound_detection.html')

def xray_pred(request):
    if request.method=="POST":
        f=request.FILES['sentFile']
        response = {}
        file_name = default_storage.save(f.name, f)
        media_dir = os.path.dirname(os.path.dirname(__file__))+'/media'
        image = load_img((os.path.join(media_dir,file_name)), target_size=(224, 224))
        image = np.array(image)/255
        image = image.reshape(-1, 224, 224, 3)

        model=load_model(os.path.dirname(__file__)+'/model.h5')
        prediction = model.predict(image)

        if prediction>0.5:
            result="COVID NOT DETECTED"
        else:
            result="COVID DETECTED"
        
        response['pred']=result

        response['pred']=result
        return render(request,'detection.html',response)
    else:
        return render(request,'detection.html')