```ts
import { createSileroVad } from "https://deno.land/x/vad";

const vad = await createSileroVad({ sampleRate: 8000 });
const audio = new Uint8Array([]);
const isSpeech = (await VAD(audio)) > 0.5;
```
