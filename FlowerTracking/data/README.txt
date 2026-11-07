Tracks_XXX.XLS contains the raw digitized data from 100 fps videos of moths tracking the robotic flower. Luminance levels are indicated in the filenames, species are abbreviated as M (Macroglossum stellatarum), Ma (Manduca sexta) and D (Deilephila elpenor). Each set of four columns is data from one video with the X,Y points for the flower and the moth respectively. Videos were either 10s or 20 seconds long for a total of 1000 or 2000 rows.



ANALYSIS_XXX.xls is a matrix containing the processed tracking data from each of the moth species (names indicated in filename). Columns are as follows:

First: An integer identifier for each individual moth (same as trial).
Second: The component frequency of the command flower movement (0.2-13.7 Hz for each moth).
Third: luminance condition (1-3, code explained in the column header).
Fourth: The gain of the moth at that frequency (a gain of 1 indicates the same amplitude tracking).
Fifth: The phase of the moth w.r.t. the flower at that frequency in radians. Data were unwrapped as described in the methods.
Sixth: Tracking error as calculated in the paper.
Seventh: The complex valued response of the moth with respect to the flower.

