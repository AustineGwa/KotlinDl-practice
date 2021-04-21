import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.Flatten
import org.jetbrains.kotlinx.dl.api.core.layer.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.jetbrains.kotlinx.dl.datasets.handlers.extractImages
import org.jetbrains.kotlinx.dl.datasets.handlers.extractLabels
import java.io.File

fun  main(){
       //defining labels
       val stringLabels = mapOf(0 to "T-shirt/top",
           1 to "Trouser",
           2 to "Pullover",
           3 to "Dress",
           4 to "Coat",
           5 to "Sandal",
           6 to "Shirt",
           7 to "Sneaker",
           8 to "Bag",
           9 to "Ankle boot"
       )

//Here's how we define a neural network that consists of a few simple layers in a sequence: (neural  network structure)
       val model = Sequential.of(
           Input(28,28,1),
           Flatten(),
           Dense(300),
           Dense(100),
           Dense(10)
       )

       model.use {
           it.compile(
               optimizer = Adam(),
               loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
               metric = Metrics.ACCURACY
           )
           // next step here is training the model: this is described in the next tutorial
           // ...

           val (train, test) = Dataset.createTrainAndTestDatasets(
               trainFeaturesPath = "datasets/mnist/train-images-idx3-ubyte.gz",
               trainLabelsPath = "datasets/mnist/train-labels-idx1-ubyte.gz",
               testFeaturesPath = "datasets/mnist/t10k-images-idx3-ubyte.gz",
               testLabelsPath = "datasets/mnist/t10k-labels-idx1-ubyte.gz",
               numClasses = 10,
               ::extractImages,
               ::extractLabels
           )

           val (newTrain, validation) = train.split(splitRatio = 0.95)

           // You can think of the training process as "fitting" the model to describe the given data :)
           it.fit(
               dataset = newTrain,
               epochs = 10,
               batchSize = 100
           )

           val accuracy = it.evaluate(dataset = validation, batchSize = 100).metrics[Metrics.ACCURACY]

           println("Accuracy: $accuracy")
           it.save(File("src/model/my_model"))


           infer()
       }

}

fun infer() {

}

