import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.datasets.handlers.extractImages
import org.jetbrains.kotlinx.dl.datasets.handlers.extractLabels
import java.io.File


fun reshapeInput(inputData: FloatArray): Array<Array<FloatArray>> {
    val reshaped = Array(
        1
    ) { Array(28) { FloatArray(28) } }
    for (i in inputData.indices) reshaped[0][i / 28][i % 28] = inputData[i]
    return reshaped
}

fun main(){
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

    val (train, test) = org.jetbrains.kotlinx.dl.datasets.Dataset.createTrainAndTestDatasets(
        trainFeaturesPath = "datasets/mnist/train-images-idx3-ubyte.gz",
        trainLabelsPath = "datasets/mnist/train-labels-idx1-ubyte.gz",
        testFeaturesPath = "datasets/mnist/t10k-images-idx3-ubyte.gz",
        testLabelsPath = "datasets/mnist/t10k-labels-idx1-ubyte.gz",
        numClasses = 10,
        ::extractImages,
        ::extractLabels
    )

    InferenceModel.load(File("src/model/my_model")).use {
        it.reshape(::reshapeInput)
        val prediction = it.predict(test.getX(0))
        val actualLabel = test.getLabel(0)

        println("Predicted label is: $prediction. This corresponds to class ${stringLabels[prediction]}.")
        println("Actual label is: $actualLabel.")
    }
}
