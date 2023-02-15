```ts
import createVad from "https://deno.land/x/vad";

const vad = await createVad({ sampleRate: 8000 });
const audio = new Uint8Array([]);
const isSpeech = (await VAD(audio)) > 0.5;
```
