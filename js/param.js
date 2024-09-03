document.addEventListener('DOMContentLoaded', main, false);

function main() {
	let svgs = document.querySelectorAll("span");

	svgs.forEach(function (svg) {
		svg.addEventListener("click", function () {
			let smiles = svg.getAttribute("smiles");
			let natoms = svg.getAttribute("natoms");
			let pid = svg.getAttribute("pid");

			// get the dialog box initially in the page
			let dialog = document.getElementById("modal-box");

			// construct a new div to go into the modal, then add in the SMILES,
			// natoms, and the svg
			let frame = document.createElement("div");
			frame.setAttribute("id", "modal-box-content");
			if (smiles) {
				let s = document.createElement("p");
				s.appendChild(document.createTextNode("SMILES: " + smiles));
				frame.setAttribute("smiles", smiles);
				frame.appendChild(s);
			}
			if (natoms) {
				let n = document.createElement("p");
				n.appendChild(document.createTextNode(natoms + " atoms"));
				frame.setAttribute("natoms", natoms);
				frame.appendChild(n);
			}
			if (pid) {
				frame.setAttribute("pid", pid);
			} else {
				console.log("WARNING: pid unset");
			}
			frame.appendChild(svg.cloneNode(true));

			// put the new frame into the dialog and display it
			let toReplace = document.getElementById("modal-box-content");
			dialog.replaceChild(frame, toReplace);
			dialog.showModal();
		});
	});
}
