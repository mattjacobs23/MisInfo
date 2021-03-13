console.log('loaded...');
const MISINFO_URL = "http://127.0.0.1:8000/api/predict-misinfo";

let spanSelection = null;

async function detectMisInfo(text) {
    const data = {
        text: text
    }
    return fetch(MISINFO_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
    });
}

document.addEventListener("mouseup", (event) => {
    if (spanSelection) {
        // Reset and remove span selection
        document.body.removeChild(spanSelection);
        spanSelection = null;
    }
    let text = ""
    if (window.getSelection) {
        text = window.getSelection().toString();
    } else if (document.selection && document.selection.type != "Control") {
        text = document.selection.createRange().text;
    }
    if (text === '') return;
    detectMisInfo(text)
        .then(res => res.json())
        .then(data => {
            const imgURL = chrome.runtime.getURL("pictures/thinking_emoji_48.png");
            const spanElem = document.createElement("span");

            spanElem.className = "popup-tag";
            spanElem.style.display = "flex";
            spanElem.style.left = `${window.scrollX + event.clientX}px`;
            spanElem.style.top = `${window.scrollY + event.clientY}px`;
            let label;
            if (!data.label) {
                label = "FALSE!";
                spanElem.style.backgroundColor = "red";
            } else {
                label = "TRUE!";
                spanElem.style.backgroundColor = "#4be371";
            }
            spanElem.innerHTML = `
                <img class="img-sty" src=${imgURL} height=32 width=32> ${label}
            `;
            document.body.appendChild(spanElem);
            spanSelection = spanElem;

        })
        .catch((error) => {
            console.error("Error:", error);
        });

});
