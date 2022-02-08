function RunPythonSim() {
    var sim_duration = document.getElementById("sim_duration");
    sim_duration = sim_duration.value;
    if (isNaN(sim_duration)) {
        document.getElementById("ticsInfo").innerHTML = "The number of tics specified incorrectly";
    }
    else {
        document.getElementById("ticsInfo").innerHTML = "Will run simulations for " + sim_duration + " tics";

        runPyScript(sim_duration);
    }

}

document.getElementById("submit_button").onclick = RunPythonSim;

// url: "http://localhost:8000/main.py"
// success: callbackFunc

/*
var jqXHR = $.ajax({
    type: "POST",
    url: "/main.py",
    data: JSON.stringify({ param: input })
});

return jqXHR.responseText;

*/


function runPyScript(input) {

    $.ajax({
        type: "POST",
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        url: "/main.py",
        data: JSON.stringify({ 'sim_dur': input }),
        success: function (response) {
            console.log(response)
        }
    });

}

