﻿using Lazar.ML.SentimentAnalysis.Models;
using Microsoft.ML;
using System;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace Lazar.ML.SentimentAnalysis
{
    public class SentimentAnalysis
    {
        private MLContext _mlContext;
        private ITransformer _model;


        public SentimentAnalysis()
        {
            _mlContext = new MLContext();
        }

        public void LoadModel(string path)
        {
            try
            {
                DataViewSchema dataSchema;
                using (var stream = System.IO.File.OpenRead(path))
                {
                    _model = _mlContext.Model.Load(stream, out dataSchema);
                }
            }
            catch (Exception ex)
            {
                throw new ApplicationException(ex.Message);
            }

        }

        public void TrainModel(string trainDataPath, string outputPath = null)
        {
            try
            {
                if (File.Exists(trainDataPath))
                {
                    IDataView dataView = _mlContext.Data.LoadFromTextFile<SentimentData>(trainDataPath, hasHeader: false);
                    TrainTestData splitDataView = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
                    var estimator = _mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.Text))
                           .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
                    _model = estimator.Fit(splitDataView.TrainSet);
                    if (outputPath != null)
                    {
                        _mlContext.Model.Save(_model, dataView.Schema, outputPath);
                    }
                }
            }
            catch (Exception ex)
            {
                throw new ApplicationException(ex.Message);
            }
        }

        public SentimentPrediction MakePrediction(SentimentData data)
        {
            try
            {
                PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = _mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(_model);
                return predictionFunction.Predict(data);
            }
            catch (Exception ex)
            {
                throw new ApplicationException(ex.Message);
            }

        }
    }
}
