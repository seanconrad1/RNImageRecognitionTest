import React, { useState, useEffect } from "react";
import {
  StyleSheet,
  Text,
  View,
  ActivityIndicator,
  StatusBar,
  Image,
  TouchableOpacity,
} from "react-native";
import * as tf from "@tensorflow/tfjs";
import { fetch } from "@tensorflow/tfjs-react-native";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as jpeg from "jpeg-js";
import * as ImagePicker from "expo-image-picker";
// import { launchCamera, launchImageLibrary } from "react-native-image-picker";
// import { Predictions } from "./types/types";
import { Buffer } from "buffer";

const App = () => {
  const [isTfReady, setIsTfReady] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [image, setImage] = useState(null);
  const [model, setModel] = useState(null);

  useEffect(() => {
    const myAsyncFunction = async () => {
      await tf.setBackend("cpu");
      await tf.ready();
      setIsTfReady(true);
      const val = await mobilenet.load();
      setModel(val);
      setIsModelReady(true);
    };
    myAsyncFunction();
  }, []);

  useEffect(() => {
    // console.log("Image was updated!");
    if (image) {
      classifyImage();
    }
  }, [image]);

  const imageToTensor = (rawImageData) => {
    const TO_UINT8ARRAY = true;
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
    // Drop the alpha channel info for mobilenet
    const buffer = new Uint8Array(width * height * 3);
    let offset = 0; // offset into original data
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];

      offset += 4;
    }

    return tf.tensor3d(buffer, [height, width, 3]);
  };

  const classifyImage = async () => {
    try {
      const imageAssetPath = Image.resolveAssetSource(image);
      const response = await fetch(imageAssetPath.uri, {}, { isBinary: true });
      const rawImageData = await response.arrayBuffer();
      const imageTensor = imageToTensor(rawImageData);
      const predictions = await model.classify(imageTensor);
      setPredictions(predictions);
      console.log(predictions);
    } catch (error) {
      console.log(error);
    }
  };

  const selectImage = async () => {
    try {
      let response = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All,
        allowsEditing: true,
        aspect: [4, 3],
      });

      if (!response.cancelled) {
        const source = { uri: response.uri };
        setImage(source);
      }
    } catch (error) {
      console.log(error);
    }
  };

  const renderPrediction = (prediction, idx: number) => {
    return (
      <View key={idx} style={styles.percentageAndName}>
        <Text style={styles.percentages}>
          {(prediction.probability * 100).toFixed(1)}%
        </Text>
        <View style={styles.predictionContainer}>
          <Text key={prediction.className} style={styles.predictions}>
            {idx + 1}: {prediction.className.split(",")[0]}
            {"\n"}
          </Text>
        </View>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <View style={styles.loadingContainer}>
        <Text style={styles.text}>
          TensorFlow Ready? {isTfReady ? <Text>✅</Text> : ""}
        </Text>

        <View style={styles.loadingModelContainer}>
          <Text style={styles.text}>Model ready? </Text>
          {isModelReady ? (
            <Text style={styles.text}>✅</Text>
          ) : (
            <ActivityIndicator size="small" />
          )}
        </View>
      </View>
      <TouchableOpacity
        style={styles.imageWrapper}
        onPress={isModelReady ? selectImage : undefined}
      >
        {image && <Image source={image!} style={styles.imageContainer} />}

        {isModelReady && !image && (
          <Text style={styles.transparentText}>Tap to choose image</Text>
        )}
      </TouchableOpacity>
      <View style={styles.predictionWrapper}>
        {isModelReady && image && (
          <Text style={styles.text}>
            Predictions: {!loading ? "" : "Predicting..."}
          </Text>
        )}
        {!loading &&
          predictions &&
          predictions?.map((p, idx: number) => renderPrediction(p, idx))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#171f24",
    alignItems: "center",
  },
  loadingContainer: {
    marginTop: 80,
    justifyContent: "center",
    textAlign: "left",
  },
  text: {
    color: "#ffffff",
    fontSize: 16,
    textAlign: "left",
  },
  loadingModelContainer: {
    flexDirection: "row",
    marginTop: 10,
  },
  imageWrapper: {
    width: 280,
    height: 280,
    padding: 10,
    borderColor: "#cf667f",
    borderWidth: 5,
    marginTop: 40,
    marginBottom: 10,
    position: "relative",
    justifyContent: "center",
    alignItems: "center",
  },
  imageContainer: {
    width: 250,
    height: 250,
    position: "absolute",
    top: 10,
    left: 10,
    bottom: 10,
    right: 10,
  },
  predictionWrapper: {
    height: 100,
    width: "100%",
    flexDirection: "column",
    alignItems: "center",
  },
  transparentText: {
    color: "#ffffff",
    opacity: 0.7,
  },

  predictionContainer: {
    width: "50%",
    borderWidth: 1,
    borderColor: "lightgrey",
    borderRadius: 100,
    margin: 10,
    padding: 20,
  },
  predictions: {
    fontSize: 16,
    textAlign: "left",
    color: "white",
    textTransform: "capitalize",
  },
  percentages: {
    color: "white",
    fontSize: 17,
  },
  percentageAndName: {
    flexDirection: "row",
    alignItems: "center",
  },
});

export default App;
