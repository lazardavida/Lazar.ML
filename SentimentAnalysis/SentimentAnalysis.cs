using Lazar.ML.SentimentAnalysis.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
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
                if (File.Exists(path))
                {
                    using (var stream = System.IO.File.OpenRead(path))
                    {
                        _model = _mlContext.Model.Load(stream, out dataSchema);
                    }
                } else
                {
                    throw new FileNotFoundException("Unable to find model to be loaded");
                }
            }
            catch (Exception ex)
            {
                throw new ApplicationException(ex.Message);
            }
        }

        public CalibratedBinaryClassificationMetrics TrainModel(string trainDataPath, string outputPath = null)
        {
            try
            {
                if (File.Exists(trainDataPath))
                {
                    var dataView = _mlContext.Data.LoadFromTextFile<SentimentData>(trainDataPath, hasHeader: false);
                    var splitDataView = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
                    var estimator = _mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.Text))
                           .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
                    _model = estimator.Fit(splitDataView.TrainSet);
                    if (outputPath != null)
                    {
                        _mlContext.Model.Save(_model, dataView.Schema, outputPath);
                    }
                    var predictions = _model.Transform(splitDataView.TestSet);
                    return _mlContext.BinaryClassification.Evaluate(predictions, "Label");
                } else
                {
                    throw new FileNotFoundException("Unable to find training data");
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

