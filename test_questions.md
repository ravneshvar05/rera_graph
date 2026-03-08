# GraphRAG System Test Questions

Here is a curated set of 20 test questions categorized by the testing criteria you specified. Each question includes the **ground truth** (the target JSON brochure containing the correct answer) so you can verify the system's responses.

## 1. Asking for suggestions from multiple different locations
**Q1.** "Can you suggest some 2 BHK apartments in Sarkhej, Vinzol, or Gamdi Gaam?"
* **Ground Truth:** `Brochure 11.json` (Sunflower Enclave - Sarkhej), [Brochure6.json](file:///d:/Rerachatbot/output/Brochure6.json) (AMBER - Sarkhej), [Brochure4.json](file:///d:/Rerachatbot/output/Brochure4.json) (OUM ORBIT - Vinzol), [Brochure1.json](file:///d:/Rerachatbot/output/Brochure1.json) (Svasaar Pravesh - Gamdi Gaam)

**Q2.** "I am looking for a 4 BHK or 5 BHK property in Gokuldham, Science City Road, or Iscon-Ambli Road. What are my options?"
* **Ground Truth:** `KP_Villa Bochure.json` (KP Villas), `Brochure 10.json` (SATYAMEV ELYSIUM), `Brochure 12.json` (Palak Elina)

**Q3.** "Suggest residential projects available in Thaltej, Paldi, or Naranpura."
* **Ground Truth:** `Brochure 13.json` (DWARKESH GREENS - Thaltej), `Brochure 14.json` (Gandhi Galaxy - Paldi), `Brochure 7.json` (SURYA KUTIR - Naranpura)

**Q4.** "Are there any apartments with a gym in Shantigram, Satellite, or New Nikol-Naroda Road?"
* **Ground Truth:** `Brochure 15.json` (Elysium at Shantigram), `Brochure 9.json` (Swastik Harmony)

---

## 2. Finding a particular house near landmarks
**Q5.** "Find me a 3 BHK apartment near Anand Niketan School in Thaltej."
* **Ground Truth:** `Brochure 13.json` (DWARKESH GREENS)

**Q6.** "I'm looking for a 2 BHK apartment close to Nirma University or Vaishno Devi Temple."
* **Ground Truth:** `Brochure 15.json` (Elysium at Shantigram)

**Q7.** "Show me properties near Kotarpur Water Works that offer 4 BHK villas."
* **Ground Truth:** `Brochure 16.json` (Sudama Homes)

**Q8.** "Is there a 2 BHK project located opposite the Karnavati APMC Market?"
* **Ground Truth:** [Brochure4.json](file:///d:/Rerachatbot/output/Brochure4.json) (OUM ORBIT)

---

## 3. Testing queries based on address parts
**Q9.** "Do you have any residential projects located on Iscon-Ambli Road?"
* **Ground Truth:** `Brochure 12.json` (Palak Elina)

**Q10.** "Can you find me an apartment situated on Shaikh Adam Abuvala Road in Paldi?"
* **Ground Truth:** `Brochure 14.json` (Gandhi Galaxy)

**Q11.** "I'm searching for a property near the 200 FT. Ring Road in the Hanspura area."
* **Ground Truth:** `Brochure 19.json` (Devnandan Sankalp City)

**Q12.** "Are there any 3 BHK options situated on Pernatirth Derasar Road in Satellite?"
* **Ground Truth:** [Brochure3.json](file:///d:/Rerachatbot/output/Brochure3.json) (RATNAAKAR 6 Beaumonde) (Note: Project has 4/5 BHK but it will test the system's spatial recall vs availability mismatch logic)

---

## 4. Testing with amenities
**Q13.** "Find me a 4 BHK villa in Isanpur that has a jogging track and is close to a BRTS station."
* **Ground Truth:** `Brochure 18.json` (NIVAAN GREENS)

**Q14.** "I need a 2 BHK apartment in Sarkhej that features a children's play area and CCTV cameras."
* **Ground Truth:** `Brochure 11.json` (Sunflower Enclave)

**Q15.** "Which 3 BHK apartments in Naranpura provide a 24-hours water supply and a park or garden?"
* **Ground Truth:** `Brochure 7.json` (SURYA KUTIR)

**Q16.** "Are there any properties designed with a piped gas facility, smart card access, and a clubhouse?"
* **Ground Truth:** `Brochure 15.json` (Elysium at Shantigram)

---

## 5. Asking different locations for a particular type of house
**Q17.** "Where in Ahmedabad can I find a 5 BHK apartment?"
* **Ground Truth:** `Brochure 12.json` (Palak Elina - Iscon-Ambli Road) and [Brochure3.json](file:///d:/Rerachatbot/output/Brochure3.json) (RATNAAKAR 6 Beaumonde - Satellite)

**Q18.** "Which locations offer 4 BHK villas?"
* **Ground Truth:** `Brochure 16.json` (Sudama Homes - Chiloda-Kotarpur), `Brochure 18.json` (NIVAAN GREENS - Isanpur), and `KP_Villa Bochure.json` (KP Villas - Gokuldham)

---

## 6. Testing fuzzy queries (typos and descriptive context)
**Q19.** "Looking for a big 3 bedder appt somewhere near torrrnt ofice in naranpura with ample parking space."
* **Ground Truth:** `Brochure 7.json` (SURYA KUTIR)

**Q20.** "Need a lavish five bedroom house with a gym and minitheatre on scienc city rd."
* **Ground Truth:** `Brochure 10.json` (SATYAMEV ELYSIUM - 5 BHK Penthouse)
