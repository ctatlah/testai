let recorder, dataChunks = [];
let player = document.getElementById("recording-audio");

function startRecording() {
	// clear previous errors and/or text
	document.getElementById("error").innerHTML = "";
	document.getElementById("error").style.visibility = "hidden";
	document.getElementById("speech-text").innerHTML = "";
	
	// toggle recoding buttons
	document.getElementById("start-recording-btn").disabled = true;
    document.getElementById("end-recording-btn").disabled = false;
      	
    // record audio from browser
 	navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
		recorder = new MediaRecorder(stream);
		recorder.start();
		
		// recording audio
		recorder.addEventListener("dataavailable", (e) => {
			dataChunks.push(e.data);
  		});
		
		// after recording is done play back recording and then
		// send it to backend server to be processed
		recorder.addEventListener("stop", () => {
  			playRecording();
  			sendToServer();
		});
	});
}

function stopRecording() {
    recorder.stop();
    document.getElementById("end-recording-btn").disabled = true;
    document.getElementById("start-recording-btn").disabled = false;
}

function playRecording() {
	let audioblob = new Blob(dataChunks);
  	let audioURL = URL.createObjectURL(audioblob);
  	const audio = new Audio(audioURL);
  	player.style.display = "block";
  	player.src = audio;
  	audio.play();
}

async function sendToServer() {
	//const blob = new Blob(dataChunks, {type: "audio/webm"} );
	//const xhr = new XMLHttpRequest();
    //xhr.open('POST', '/speechin', true);
    //xhr.onload = function(e) {
	//	const responseMsg = JSON.parse(xhr.responseText);
	//	if (xhr.status === 200) {
	//		document.getElementById("speech-text").innerHTML = responseMsg.message;
	//	} else {
	//		errorMsg = "Error: " + xhr.statusText + "<br>Details: " + responseMsg.error;
	//		document.getElementById("error").innerHTML = errorMsg;
	//		document.getElementById("error").style.visibility = "visible";
	//	}
    //};
    //xhr.onerror = function(e) {
	//	console.error("Error sending audio:", e);
    //};
    //xhr.send(blob);
    
    const blob = new Blob(dataChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append("audio_data", blob);

    try {
      	const response = await fetch('/speechin', {
        	method: 'POST',
        	body: formData
      	});
      	const responseText = await response.text();
      	const responseMsg = JSON.parse(responseText);
      	outputMsg = "Transcription:<br>" + responseMsg.message;
      	document.getElementById("speech-text").innerHTML = outputMsg;
    } catch (error) {
		document.getElementById("error").innerHTML = errorMsg;
		document.getElementById("error").style.visibility = "visible";
    }

    dataChunks = [];
}

document.getElementById("start-recording-btn").addEventListener("click", startRecording);
document.getElementById("end-recording-btn").addEventListener("click", stopRecording);
document.getElementById("recording-audio").addEventListener("loadeddata", playRecording);