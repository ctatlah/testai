function sendMessage() {
	// Clear and hide any previous errors and statuses
	document.getElementById("error").innerHTML = ""
	document.getElementById("error").style.visibility = "hidden";
	document.getElementById("status").innerHTML = ""
	document.getElementById("status").style.visibility = "hidden";
	
	// Get user question
	const userInput = document.getElementById("user-input").value;
    document.getElementById("status").innerHTML = "Thinking..."
	document.getElementById("status").style.visibility = "visible";

    // Post XMLHttpRequest to python file with user's input
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/MathTutorWeb");
    xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    xhr.send("message=" + encodeURIComponent(userInput));
    
    // Handle the response from the python script
    xhr.onload = function() {
		// Update status
		document.getElementById("status").innerHTML = "Done!"
		document.getElementById("status").style.visibility = "visible";
		
		const obj = JSON.parse(xhr.responseText);
		if (xhr.status === 200) {
			// Success show response
			document.getElementById("answer").innerHTML = obj.message;
        } else {
			// Something went wrong what did the server say?
			errorMsg = "Error: " + xhr.statusText + "<br>Details: " + obj.error;
          	document.getElementById("error").innerHTML = errorMsg;
          	document.getElementById("error").style.visibility = "visible";
        }
   
    };
    
	// Clear the user input field
	document.getElementById("user-input").value = " ";
}