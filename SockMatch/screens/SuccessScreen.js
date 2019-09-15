import React from 'react';
import { StyleSheet, View, Text, TouchableOpacity } from 'react-native';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import { faCheck, faTimes, faCog } from '@fortawesome/free-solid-svg-icons';;


export default class CameraScreen extends React.Component {
  
  state = {
    matches: 'loading',
    type1: 'Loading..',
    type2: 'Loading..',
  }

  componentWillMount() {
    this.callServer();
  }
 
  static navigationOptions = {  
    title: 'Sock Match',
    headerTitleStyle: {
      color: '#FAC644',
      fontSize: 24,
    },
    headerStyle: {
      backgroundColor: '#01507B',
      borderWidth: 0,
    },
    headerLeft: null
  };


  callServer = () => {
    fetch('http://104.197.104.75:5000', {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sock1: this.props.navigation.state.params.sock1,
        sock2: this.props.navigation.state.params.sock2,
      }),
    })
    .then((response) => response.json())    
    .then((responseJson) => {
      this.setState(responseJson);
    })
    .catch((error) => {
      console.error(error);
    })
  }
  
  render() {  
    if (this.state.matches == "loading") {
      var status = 'Loading';
      var backgroundC = '#add8e6';
      var textColor = 'blue'
      var icon = <FontAwesomeIcon color={'blue'} size={ 150 } style={{color:'black', fontSize: 36, alignSelf: 'center'}} icon={faCog}/> 
        
    } else if (this.state.matches == true) {
      var status = 'Match';
      var backgroundC = '#b3ffb3';
      var textColor = 'green'
      var icon = <FontAwesomeIcon color={'green'} size={ 150 } style={{color:'black', fontSize: 36, alignSelf: 'center'}} icon={faCheck}/> 
    } else {
      var status = 'No Match';
      var backgroundC = '#ff9999';
      var textColor = 'red'
      var icon = <FontAwesomeIcon color={'red'} size={ 150 } style={{color:'black', fontSize: 36, alignSelf: 'center'}} icon={faTimes}/>
    }

    return (
      <View style={{flex: 1, backgroundColor: backgroundC}} >
        <Text style={{color: textColor, alignSelf: 'center', fontSize: 50, marginTop: 40}}>
          {status}
        </Text>
        <Text style={{alignSelf: 'center', alignItems: 'center'}}>
        {icon}
        </Text>
        <View style={styles.infoContainer}>
          <Text style={styles.info} >{'Sock1 Description: ' + this.state.type1}</Text>
          <Text style={styles.info} >{'Sock2 Description: ' + this.state.type2}</Text>
        </View>
        <TouchableOpacity
          style={styles.button}
          type="outline"
          onPress={() => this.props.navigation.goBack()}
        >
          <Text style={{color: '#FAC644', alignSelf: 'center', fontSize: 24}}>Take Another Picture</Text>
        </TouchableOpacity>
      </View>
    );
  }
}





const styles = StyleSheet.create({ 
  info: {
    fontSize: 24,
    color: 'black'
  },
  statusTitle_green: {
    color: 'green',
    alignSelf: 'center'
  },
  statusTitle_red: {
    color: 'red',
    alignSelf: 'center'
  },
  infoContainer: {
    alignSelf: 'center',
  },
  button: {
    position: 'absolute',
    bottom: 60,
    justifyContent: 'center', 
    backgroundColor: '#01507B',
    fontSize: 32,
    height: 75,
    width: 300,
    alignSelf: 'center',
    alignItems: 'center',
  }
  
});


