import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Icon } from 'react-native';
import * as Permissions from 'expo-permissions';
import { Camera } from 'expo-camera';
import * as FileSystem from 'expo-file-system';


export default class CameraScreen extends React.Component {
  static navigationOptions = {  
    title: 'Sock Match',
    headerTitleStyle: {
      color: '#FAC644',
      fontSize: 24,
    },
    headerStyle: {
      backgroundColor: '#01507B',
      borderWidth: 0,
    }
  };
  
  state = {
    hasCameraPermission: null,
    type: Camera.Constants.Type.back,
    sock1: null,
    sock2: null,
    processing: false,
  };

  snap = async () => {
    if (this.camera) {
      this.setState({processing: true});
      let uri = this.camera.takePictureAsync({
        base64: true
      }).then(data => {
          if(!this.state.sock1) {
            this.setState({sock1: data.base64});
            this.setState({processing: false});
          } else {
            this.setState({sock2: data.base64});
            this.setState({processing: false});
            this.submitData();
          }
      }).catch(err => {
        console.log("err", err);
      })
    }
  }


  submitData = () => {
    this.props.navigation.navigate('Success', {sock1: this.state.sock1, sock2: this.state.sock2});
    this.setState({sock1: null, sock2: null});
  }

  async componentDidMount() {
    const { status } = await Permissions.askAsync(Permissions.CAMERA);
    this.setState({ hasCameraPermission: status === 'granted' });
  }

  render() {
    const { hasCameraPermission } = this.state;
    if (hasCameraPermission === null) {
      return <View />;
    } else if (hasCameraPermission === false) {
      return <Text>No access to camera</Text>;
    } else {
      return (
        <View style={{ flex: 1}}>
          <Camera 
            style={{ flex: 1 }} 
            type={this.state.type}   
            ref={ref => {
              this.camera = ref;
            }}
          >
          </Camera>

          <TouchableOpacity disabled={this.state.processing}
            style={styles.circleButton}
            onPress={() => this.snap()}
            underlayColor='black'
          >
            <View style={styles.circle}></View>
          </TouchableOpacity>


        </View>
      );
    }
  }
}

const styles = StyleSheet.create({
  circleButton:{
    position: 'absolute',
    height: 100,
    width: 100,
    justifyContent:'center',
    alignSelf:'center',
    alignItems:'center',
    borderRadius: 50,
    backgroundColor:'white',
    bottom: 30,
  },
  circle: {
    backgroundColor:'white',
    borderWidth:2,
    borderColor:'grey',
    height: 84,
    width: 84,
    borderRadius: 42,
  }
});