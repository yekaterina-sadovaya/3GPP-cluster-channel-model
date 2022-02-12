
function RunPythonSim() {
	
    var car_freq = document.getElementById("car_freq");
    car_freq = car_freq.value;
	var num_clusters = document.getElementById("num_clusters");
    num_clusters = num_clusters.value;
	var num_rays = document.getElementById("num_rays");
    num_rays = num_rays.value;
	var rx_pos = document.getElementById("rx_pos");
    rx_pos = rx_pos.value;
	var tx_pos = document.getElementById("tx_pos");
    tx_pos = tx_pos.value;
	
	console.log(car_freq, typeof car_freq)
	console.log(num_clusters, typeof num_clusters )
	console.log(num_rays, typeof num_rays )
	console.log(rx_pos, typeof rx_pos )
	console.log(tx_pos, typeof tx_pos )
	
    if ((isNaN(car_freq)) || (isNaN(num_clusters)) || (isNaN(num_rays)) || (rx_pos == "") || (tx_pos == "")) {
        document.getElementById("ticsInfo").innerHTML = "All fields should be filled with numerical values";
    }
    else {
		car_freq = Number(car_freq)
		rx_pos = rx_pos.split(",").filter(x => x.trim().length && !isNaN(x)).map(Number)
		tx_pos = tx_pos.split(",").filter(x => x.trim().length && !isNaN(x)).map(Number)
		if ((car_freq > 100) && (car_freq < 0.5)) {
			document.getElementById("ticsInfo").innerHTML = "Carrier frequency should fall in the range of [0.5, 100] GHz";			
		}
		else if ((rx_pos.length != 3) || (tx_pos.length != 3)){
			document.getElementById("ticsInfo").innerHTML = "All 3 coordinates should be given and separated with commas";	
		}
		else{
			document.getElementById("ticsInfo").innerHTML = "Simulation is running, wait a moment";
			
			var input_data = new Object();
			input_data.car_freq = car_freq;
			input_data.num_clusters  = Number(num_clusters);
			input_data.num_rays = Number(num_rays);
			input_data.rx_pos = rx_pos;
			input_data.tx_pos = tx_pos;
				
			sendRequest(input_data);
		}

    }

}

document.getElementById("submit_button").onclick = RunPythonSim;


function sendRequest(input) {

    $.ajax({
        type: "POST",
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        url: "/main.py",
        data: JSON.stringify(input),
        success: function () {

            $("#includedContent").load('/figure.html'); 
        }
    });

}

