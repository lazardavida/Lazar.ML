# Lazar.ML
Sample Usage:

### initialize
`var sa = new SentimentAnalysisEngine();`

### train the model
Train without saving the model <br />
`sa.TrainModel(Path.Combine(Environment.CurrentDirectory, "sentiment-train-data.csv"));`

Train with saving the model <br />
`sa.TrainModel(Path.Combine(Environment.CurrentDirectory, "sentiment-train-data.csv"), Path.Combine(Environment.CurrentDirectory, "model.zip"));`

### or load a saved model
`sa.LoadModel(Path.Combine(Environment.CurrentDirectory, "model.zip"));`

### create the SentimentData class object to test against the model
`SentimentData data = new SentimentData { Text = "this was the worst decision ever" };`

### make the prediction
`var prediction = sa.MakePrediction(data);`

### output
`Console.WriteLine("Use text: {0}", data.Text);`<br />
`Console.WriteLine("Prediciton of {0} with probability {1}", prediction.Prediction, prediction.Probability);`
