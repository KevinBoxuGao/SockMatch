import {createAppContainer} from 'react-navigation';
import {createStackNavigator} from 'react-navigation-stack';
import CameraScreen from './screens/CameraScreen';
import SuccessScreen from './screens/SuccessScreen';


const MainNavigator = createStackNavigator({
  Camera: {screen: CameraScreen},
  Success: {screen: SuccessScreen},
});

const App = createAppContainer(MainNavigator);

export default App;