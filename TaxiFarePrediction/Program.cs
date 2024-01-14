using Microsoft.ML;
using System;
using System.IO;
using TaxiFarePrediction;

namespace TaxiFarePrediction {
    class Program
    {

       static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
       static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
       static readonly  string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {

            Console.WriteLine(Environment.CurrentDirectory);
            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _trainDataPath);
            Evaluate(mlContext, model);
            TestSinglePrediction(mlContext, model);
        }

        /*
         * Train Function
          - Loads the data
          - Extracts and Transforms the data 
          - Trains the model 
          - Returns the model 
         */
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                                    .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                                    .Append(mlContext.Regression.Trainers.FastTree());
            var model = pipeline.Fit(dataView);
            return model;
        }
       public static void Evaluate(MLContext mlContext, ITransformer model)
        {
            /*
             - Load the test dataset
             - Create the regression evaluator
             - Evaluates the model and creates metrics
             - Displays the metrics
             */
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine();
            Console.WriteLine($"************************");
            Console.WriteLine($"*       Model Quality metrics evaluation        ");
            Console.WriteLine($"*--------------------------");
            // Rsquared between 0 and 1. closer to 1, the better the model
            Console.WriteLine($"*       Rsquared Score:     {metrics.RSquared:0.##}");
            //RMS the lower the better
            Console.WriteLine($"*       Root Mean Squared Error:    {metrics.RootMeanSquaredError:#.##}");
        }
       public static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            /*
            Creates a single comment of test data
            Predicts Fare amount based on test data
            Combines test data and predictions for reporting
            Displays the predicted results
             */
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To Predict. Actual/Observed = 15.5
            };
            var prediction = predictionFunction.Predict(taxiTripSample);
            Console.WriteLine($"*********************************");
            Console.WriteLine($"Predicted Fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"*********************************");
        }
    }

}