Put rubberband.exe here.

speedman needs the Rubber Band command-line utility. It is the only engine that
can apply a non-uniform time map, which is the entire feature.

Rather than editing your Windows PATH, you can just drop the binary in this
folder and speedman will find it.

  1. Download the command-line utility (a zip file) from
     https://breakfastquay.com/rubberband/
  2. Open the zip and find rubberband.exe inside it
  3. Copy rubberband.exe into this folder
  4. Run:  speedman doctor

Rubber Band is separate software by Breakfast Quay, distributed under the GNU
General Public License. It is not bundled here -- you download it yourself.

On macOS or Linux you do not need this folder: use `brew install rubberband`
or `apt install rubberband-cli` and it will be found on PATH automatically.
