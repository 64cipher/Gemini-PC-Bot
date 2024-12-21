import google.generativeai as genai
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController
import time
import mss
import base64
import os
from PIL import Image
import io
import numpy as np
import torch
import json
import argparse
import speech_recognition as sr
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog
from ttkthemes import ThemedTk
import threading
import pyperclip  # Import de pyperclip

# Définition des constantes de configuration
# Durée de la pause entre chaque action en secondes
PAUSE_DURATION = 1
API_KEY_FILE = "api_key.txt"  # Nom du fichier de sauvegarde de la clé api
MIN_PAUSE_DURATION = 1
MAX_RETRIES = 0  # nombre maximal de tentatives d'execution
DEFAULT_MODEL = "gemini-2.0-flash-exp" # modèle par défaut
AVAILABLE_MODELS = ["gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp-1219", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b", "text-embedding-004"] # Modèle disponible dans le menu déroulant

# Couleurs et polices pour un thème plus doux
BG_COLOR = "#f0f0f0"  # Gris très clair pour le fond
TEXT_COLOR = "#333333"  # Gris foncé pour le texte
BUTTON_COLOR = "#a0d468"  # Vert doux pour les boutons
BUTTON_TEXT_COLOR = "#000000"  # Noir pour le texte des boutons
FONT_FAMILY = "Arial Rounded MT Bold"  # Police arrondie
FONT_SIZE = 11


class TaskAutomator:
    def __init__(self, api_key, output_text_widget, status_label, send_button, stop_button, model_name=DEFAULT_MODEL, max_retries = MAX_RETRIES):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.output_text_widget = output_text_widget  # Widget ou les textes doivent etre affichés
        self.status_label = status_label  # Label pour indiquer si Gemini est en cours
        self.send_button = send_button  # Bouton envoyer pour le désactiver pendant l'analyse
        self.stop_button = stop_button  # Bouton pour arrêter l'exécution
        self._stop_requested = False  # Flag pour interrompre l'exécution
        self._history = []
        self.current_instruction = None # Mémorise l'instruction courante
        self.max_retries = max_retries # Nombre maximal de tentatives

    def _log_message(self, message):
        """Affiche le message dans la zone de texte."""
        self.output_text_widget.insert(tk.END, message + "\n")
        self.output_text_widget.see(tk.END)  # Pour que la dernière ligne soit visible.

    def _capture_screen(self):
        """Capture une partie de l'écran et la retourne en base64."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            bbox = (monitor["left"], monitor["top"], monitor["width"], monitor["height"])
            sct_img = sct.grab(bbox)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _analyze_image_with_gemini_vision(self, image_base64):
        """Analyse l'image avec l'API Gemini Vision."""

        try:
            contents = [
                "Analyse l'image et détecte tous les éléments de l'interface graphique, et leurs textes",
                {"mime_type": "image/png", "data": image_base64}
            ]
            response = self.model.generate_content(contents=contents)

            if response.text:
                try:
                    data = json.loads(response.text)
                    self._log_message(f"Gemini Vision Response (parsed):\n{json.dumps(data, indent=2)}")
                    return data
                except json.JSONDecodeError as e:
                    self._log_message(f"Erreur lors du parsing JSON : {e}. La réponse brute est : {response.text}")
                    return {}
            else:
                self._log_message("L'API Gemini n'a retourné aucune réponse")
                return {}

        except Exception as e:
            self._log_message(f"Erreur lors de l'analyse avec Gemini Vision: {e}")
            return {}
    def _calculate_center(self, element):
            """Calcule le centre d'un élément à partir de ses coordonnées."""
            if "bounding_box" in element:
                x1 = element["bounding_box"]["x1"]
                y1 = element["bounding_box"]["y1"]
                x2 = element["bounding_box"]["x2"]
                y2 = element["bounding_box"]["y2"]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                return center_x, center_y
            return None, None

    def _parse_instruction(self, instruction, image_base64, retry_message=None):
        """Utilise Gemini pour analyser l'instruction, l'image et les données de vision et retourner des actions sous forme textuelle."""
        vision_data = self._analyze_image_with_gemini_vision(image_base64)  # données de vision

        prompt = f"""
            Tu es un assistant expert en automatisation d'interface graphique. Ton but est d'exécuter une instruction en interagissant avec l'interface.
            Voici l'instruction: {instruction}

            Voici les données de vision (format json):
            {vision_data}

            Tu dois retourner une liste d'actions textuelles, une action par ligne. 
            Priorise toujours les interactions avec l'interface (clics, mouvements de souris) avant la frappe au clavier.
            Si tu dois lancer un programme, ouvre le menu Démarrer, utilise type_text pour taper le nom du programme, puis appuie sur la touche enter pour valider.
            Après avoir lancé le programme, interagit directement avec son interface pour faire ce que l'on te demande.
            Une fois ta tâche terminée, ne fait rien de plus.
            Rajoute une pause après avoir interagi avec l'interface graphique, surtout si tu viens d'ouvrir une application ou un menu.
            Ajoute une pause après une action `press_key enter`.
            Quand tu dois cliquer sur un élément, utilise les données de vision pour déterminer les coordonnées précises de l'élément et clique au centre de celui-ci.
            
            Les types d'actions possibles sont:
                - 'move_mouse x y': Déplace le curseur.
                - 'click_mouse button': Simule un clic de souris. button peut être "left" ou "right". Si tu vois un élément de l'interface qui semble cliquable, utilise `click_mouse` et cible le centre de cet élément.
                - 'press_key key': Simule l'appui sur une touche du clavier. Utilise `press_key` pour les touches spéciales comme `enter`, `esc` ou `cmd`. Utilise 'cmd' pour la touche windows.
                - 'type_text text': Simule la frappe de texte. Utilise `type_text` uniquement dans les champs de texte ou pour saisir du texte libre.
                - 'wait seconds': Mets le programme en pause. Tu dois choisir la durée de la pause (en secondes) en fonction du contexte. 
                - 'capture_screen': Prend une capture d'écran.
            
            Utilise 'capture_screen' quand :
                - Tu as exécuté des actions et tu veux vérifier le résultat
                - Tu ne comprends pas l'état de l'interface
                - Tu soupçonnes un problème

            Si l'action précédente n'a pas fonctionné, explique brièvement pourquoi et prends une capture d'écran pour réévaluer la situation.
            Retourne les actions une par ligne. Si tu ne peux pas determiner les actions à faire, ne retourne rien.

            Par exemple, si tu dois cliquer sur un bouton qui contient le texte "Ouvrir", tu dois retourner l'action 'click_mouse left' et cibler le centre du bouton.
            Si tu dois écrire ton nom dans un champ texte, tu dois utiliser l'action 'type_text <ton nom>'.
            Si tu dois lancer un programme, ouvre le menu Démarrer en appuyant sur la touche windows, tape le nom du programme avec type_text puis valides avec la touche enter.

            Explique ton raisonnement avant de prendre une capture d'écran.
        """
        if retry_message:
            prompt += f"\n L'action précédente n'a pas fonctionné, voici l'erreur: {retry_message}. Essaye à nouveau."

        try:
            contents = [
                prompt,
                {"mime_type": "image/png", "data": image_base64}
            ]

            response = self.model.generate_content(contents=contents)
            self._log_message(f"Gemini Response (raw):\n{response.text}")
            # On utilise strip pour retirer les \n en debut et fin de chaine
            actions_text = response.text.strip()
        except Exception as e:
            self._log_message(f"Erreur lors de l'analyse avec Gemini: {e}")
            return []

        actions = self.parse_text_actions(actions_text, vision_data)
        self._log_message(f"Gemini Response (parsed):\n{actions}")
        return actions

    def parse_text_actions(self, actions_text, vision_data):
        """Parse les actions textuelles retournées par Gemini en une liste de dictionnaires."""
        actions = []
        if actions_text:
            for line in actions_text.strip().split("\n"):
                parts = line.strip().split()
                if not parts:
                    continue  # Ignore les lignes vides
                action_type = parts[0]
                if action_type == "move_mouse":
                    if len(parts) == 3:
                        try:
                            actions.append({"action": "mouse_move", "x": int(parts[1]), "y": int(parts[2])})
                        except ValueError:
                            self._log_message(
                                f"Erreur de parsing pour move_mouse : les coordonnées doivent être des entiers.")
                    else:
                        self._log_message(f"Erreur de parsing pour move_mouse : nombre d'arguments incorrect.")
                elif action_type == "click_mouse":
                    if len(parts) == 2:
                        # on itère sur vision data pour trouver l'element qui correspond
                        if vision_data and 'elements' in vision_data:
                          # si la réponse contient des éléments, on itère sur chaque élément pour trouver le centre.
                           found_element = False
                           for element in vision_data["elements"]:
                             if element and "text" in element and parts[1] in element["text"] :
                               center_x, center_y = self._calculate_center(element)
                               if center_x is not None and center_y is not None:
                                  actions.append({"action": "mouse_move", "x": center_x, "y": center_y})
                                  actions.append({"action": "mouse_click", "button": "left"}) # On clique toujours à gauche, on verra pour faire autrement après
                                  found_element = True
                                  break
                           if not found_element: # Si on n'a pas trouvé d'élément, on fait un simple click
                                 actions.append({"action": "mouse_click", "button": parts[1]})
                        else:
                              actions.append({"action": "mouse_click", "button": parts[1]})
                    else:
                        self._log_message(f"Erreur de parsing pour click_mouse : nombre d'arguments incorrect.")
                elif action_type == "press_key":
                  if len(parts) == 2:
                      key_name = parts[1]
                      if key_name == 'windows' or key_name == 'win' or key_name == 'cmd':
                           key = Key.cmd
                      else:
                          try:
                              key = getattr(Key, key_name) if hasattr(Key, key_name) else key_name
                          except AttributeError:
                            self._log_message(f"Erreur de parsing pour press_key, touche inconnue: {key_name}")
                            continue # On ignore les touches non reconnues
                      actions.append({"action": "keyboard_press", "key": key})
                  else:
                     self._log_message(f"Erreur de parsing pour press_key : nombre d'arguments incorrect.")
                elif action_type == "type_text":
                    if len(parts) >= 2:
                        text = " ".join(parts[1:])
                        actions.append({"action": "keyboard_type", "text": text})
                    else:
                        self._log_message(f"Erreur de parsing pour type_text : nombre d'arguments incorrect.")
                elif action_type == "wait":
                    if len(parts) == 2:
                        try:
                            actions.append({"action": "wait", "seconds": float(parts[1])})
                        except ValueError:
                            self._log_message(f"Erreur de parsing pour wait : les secondes doivent être un nombre.")
                    else:
                        self._log_message(f"Erreur de parsing pour wait : nombre d'arguments incorrect.")
                elif action_type == "capture_screen":
                        actions.append({"action": "capture_screen"})
                else:
                    self._log_message(f"Action inconnue ignorée : {action_type}")

        return actions

    def execute_actions(self, commands):
        """Exécute la liste de commandes."""
        retry_count = 0
        image_base64 = None
        error = None
        success = False # On ajoute cette variable
        while retry_count <= MAX_RETRIES and not success: # On ajoute success ici
            image_base64 = None # On réinitialise la variable ici
            for i, command in enumerate(commands):
                if self._stop_requested:
                    self._log_message("Execution interrompue.")
                    return

                self._log_message(f"Executing command: {command}")
                try:
                     if command["action"] == "mouse_move":
                        self.mouse.position = (command["x"], command["y"])
                     elif command["action"] == "mouse_click":
                        button = Button.left if command["button"] == "left" else Button.right
                        self.mouse.click(button)
                     elif command["action"] == "keyboard_press":
                        self.keyboard.press(command['key'])
                        self.keyboard.release(command['key'])
                     elif command["action"] == "keyboard_type":
                        text = command["text"]
                        for char in text:
                            self.keyboard.type(char)
                            time.sleep(0.03)  # Délai de 30 ms entre chaque caractère
                        # Option alternative : Utiliser le presse-papier
                        # pyperclip.copy(text)
                        # self.keyboard.press(Key.ctrl_l)
                        # self.keyboard.press('v')
                        # self.keyboard.release('v')
                        # self.keyboard.release(Key.ctrl_l)
                     elif command["action"] == "wait":
                        self._log_message(f"Attente de {command['seconds']} secondes")
                        time.sleep(command["seconds"]) # On garde la pause que Gemini a retourné
                     elif command["action"] == "capture_screen":
                        image_base64 = self._capture_screen()
                        self._log_message("Capture d'écran prise.")

                        if i == len(commands) - 1:
                            vision_result = self._analyze_image_with_gemini_vision(image_base64)
                            if not vision_result:
                                retry_count += 1
                                continue

                            error = self._check_action_with_gemini(vision_result)
                            if error:
                                retry_count += 1
                                self._log_message(f"L'action n'a pas fonctionnée. Tentative #{retry_count}. Erreur: {error}")
                                break
                            else:
                                success = True # Si c'est la dernière action et qu'il n'y a pas d'erreur, on passe success à true
                        else:
                            vision_result = self._analyze_image_with_gemini_vision(image_base64)
                            if not vision_result:
                                retry_count += 1
                                continue

                            error = self._check_action_with_gemini(vision_result)
                            if error:
                                retry_count += 1
                                self._log_message(f"L'action n'a pas fonctionnée. Tentative #{retry_count}. Erreur: {error}")
                                break
                except Exception as e:
                    self._log_message(f"Une erreur innatendue est survenue lors de l'execution de la commande {command}. Erreur: {e}")
                    retry_count += 1 # On augmente le nombre de tentatives
                    image_base64 = self._capture_screen()  # On prend une nouvelle capture d'écran
                    error = f"Une erreur inattendue est survenue. Erreur: {e}" # on sauvegarde l'erreur
                    break  # On sort de la boucle for pour réanalyser

            else:
                if not image_base64:
                    success = True # On passe success à True si toutes les actions ont été faite et qu'il n'y a pas de capture d'écran à la fin
                
            if image_base64:
                commands = self._parse_instruction(self.current_instruction, image_base64, f"L'action précédente n'a pas fonctionné. Tentative #{retry_count}. Erreur: {error}")
                if not commands:
                    self._log_message(f"Gemini n'a pas retourné de nouvelle action.")
                    return
            else:
                return  # Pas de capture d'écran.
        self._log_message(f"L'action n'a pas fonctionnée après {MAX_RETRIES} tentatives.")


    def _check_action_with_gemini(self, vision_data):
        """Utilise Gemini pour vérifier si l'action a fonctionné."""
        prompt = f"""
          Voici l'instruction qui a été exécutée: {self.current_instruction}
          Voici les informations sur l'interface graphique après execution:
          {vision_data}

          Dis moi si l'action a bien fonctionnée ou non. Si ce n'est pas le cas donne moi une raison de l'échec, sinon ne dis rien.
          Réponds par du texte uniquement, ne fait pas de code ou de json.
        """
        try:
            response = self.model.generate_content(prompt)
            if response.text:
                 self._log_message(f"Gemini a répondu à la vérification: {response.text}")
                # On utilise strip pour retirer les \n en debut et fin de chaine
                 return response.text.strip()  # Renvoi la réponse si Gemini a détecté une erreur
            else:
                return None
        except Exception as e:
            self._log_message(f"Erreur lors de la vérification de l'action avec Gemini: {e}")
            return None
    
    def _add_to_history(self, instruction, image_base64):
        """Ajoute l'instruction et la capture à l'historique."""
        self._history.append({
            "instruction": instruction,
            "image_base64": image_base64
        })

    def set_status(self, status):
        """Met à jour le label de statut dans l'interface graphique."""
        self.status_label.config(text=status)
        self.status_label.update_idletasks()  # Force l'update du label

    def set_send_button_state(self, state):
        """Active ou désactive le bouton Envoyer."""
        self.send_button.config(state=state)
        self.send_button.update_idletasks()  # Force l'update du bouton

    def set_stop_button_state(self, state):
        """Active ou désactive le bouton Interrompre"""
        self.stop_button.config(state=state)
        self.stop_button.update_idletasks()

    def request_stop(self):
        """ Demande l'arrêt de l'exécution """
        self._stop_requested = True
        self._log_message("Demande d'interruption en cours...")
        self.set_stop_button_state(tk.DISABLED)

    def run(self, instruction):
        """Analyse l'instruction et l'image, puis exécute les commandes."""
        self._stop_requested = False  # Reset le flag avant d'executer
        self.set_stop_button_state(tk.NORMAL)  # Réactiver le bouton au début de l'analyse
        self.current_instruction = instruction # On sauvegarde l'instruction
        thread = threading.Thread(target=self._run_in_thread, args=(instruction,))
        thread.start()
        self._history = []  # Réinitialise l'historique à chaque run

    def _run_in_thread(self, instruction):
        """Analyse l'instruction et l'image, puis exécute les commandes (dans un nouveau thread)."""
        self.set_send_button_state(tk.DISABLED)  # Désactiver le bouton
        self.set_status("Gemini réfléchi...")
        image_base64 = self._capture_screen()
        self._add_to_history(instruction, image_base64) # Ajouter la capture à l'historique
        commands = self._parse_instruction(instruction, image_base64)
        if commands:
            self.execute_actions(commands)
        self.set_status("Gemini est prêt.")
        self.set_send_button_state(tk.NORMAL)  # Réactiver le bouton
        self.set_stop_button_state(tk.DISABLED)  # Désactiver le bouton interrompre

    def _recognize_speech(self, callback):
        """Reconnaît la parole via le micro et la transforme en texte (dans un nouveau thread)."""
        thread = threading.Thread(target=self._recognize_speech_in_thread, args=(callback,))
        thread.start()

    def _recognize_speech_in_thread(self, callback):
        """Reconnaît la parole via le micro et la transforme en texte (dans un nouveau thread)."""
        r = sr.Recognizer()
        with sr.Microphone() as source:
            self._log_message("Parlez...")
            try:
                audio = r.listen(source, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                self._log_message("Aucune parole détectée, veuillez réessayer.")
                callback(None)
                return

            try:
                self._log_message("Reconnaissance en cours...")
                text = r.recognize_google(audio, language="fr-FR")
                self._log_message(f"Vous avez dit : {text}")
                callback(text)
            except sr.UnknownValueError:
                self._log_message("Impossible de comprendre l'audio")
                callback(None)
            except sr.RequestError as e:
                self._log_message(f"Erreur lors de la requête de reconnaissance vocale: {e}")
                callback(None)


def load_api_key():
    """Charge la clé API depuis le fichier de sauvegarde."""
    if os.path.exists(API_KEY_FILE):
        try:
            with open(API_KEY_FILE, "r") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Erreur lors du chargement de la clé API depuis le fichier : {e}")
            return None
    return None


def save_api_key(api_key):
    """Sauvegarde la clé API dans le fichier de sauvegarde."""
    try:
        with open(API_KEY_FILE, "w") as f:
            f.write(api_key)
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de la clé API : {e}")
        return False


def main():
    """Fonction principale pour la création de l'interface graphique."""
    # Chargement de la clé API depuis le fichier
    api_key = load_api_key()

    # Si la clé API n'est pas trouvée, la demander à l'utilisateur et la sauvegarder
    if not api_key:
        root = ThemedTk(theme="equilux")  # Fenetre avec le thème par défaut
        root.withdraw()
        api_key = simpledialog.askstring("Clé API", "Veuillez entrer votre clé API Gemini :")
        root.destroy()

        if not api_key:
            print("Vous devez fournir une clé API.")
            exit()
        if not save_api_key(api_key):
            print("Erreur lors de la sauvegarde de la clé API")
            exit()

    root = ThemedTk(theme="plastik")  # On garde un thème sombre car le gris sur du blanc est peut etre pas assez contrasté pour être lisible
    root.title("Gemini PC Control")

    # Icône de la fenêtre
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "gemini_icon.png")
        icon = tk.PhotoImage(file=icon_path)
        root.iconphoto(True, icon)
    except Exception as e:
        print(f"Erreur lors du chargement de l'icone : {e}")

    # Modification du style de base
    root.configure(bg=BG_COLOR)

    # Style pour le frame
    style = ttk.Style()
    style.configure("TFrame", background=BG_COLOR)

    # Création d'une zone de texte pour la sortie
    output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg="white", fg=TEXT_COLOR,
                                           font=(FONT_FAMILY, FONT_SIZE))
    output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Label pour indiquer le statut
    status_label = ttk.Label(root, text="Gemini est prêt.", foreground=TEXT_COLOR, background=BG_COLOR,
                             font=(FONT_FAMILY, FONT_SIZE))
    status_label.pack(pady=(0, 5))

    # Champ de saisie pour les instructions
    input_frame = ttk.Frame(root, style="TFrame")  # Utiliser le style
    input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

    input_entry = ttk.Entry(input_frame, font=(FONT_FAMILY, FONT_SIZE))
    input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def send_instruction():
        instruction = input_entry.get()
        if instruction:
            output_text.insert(tk.END, f"Tâche demandée: {instruction}\n")
            output_text.see(tk.END)
            input_entry.delete(0, tk.END)
            automator.run(instruction)
        else:
            output_text.insert(tk.END, "Veuillez entrer une instruction\n")
            output_text.see(tk.END)

    def use_voice_command():
        def voice_callback(instruction):
            if instruction:
                output_text.insert(tk.END, f"Tâche demandée: {instruction}\n")
                output_text.see(tk.END)
                automator.run(instruction)

        automator._recognize_speech(voice_callback)

    # Bouton Envoyer
    send_button = ttk.Button(
        input_frame,
        text="Envoyer",
        command=send_instruction,
        style="Custom.TButton"  # Style personnalisé
    )
    send_button.pack(side=tk.LEFT, padx=5)

    # Bouton Micro
    mic_button = ttk.Button(
        input_frame,
        text="Micro",
        command=use_voice_command,
        style="Custom.TButton"  # Style personnalisé
    )
    mic_button.pack(side=tk.LEFT, padx=5)

    # Bouton Interrompre
    stop_button = ttk.Button(
        input_frame,
        text="Interrompre",
        command=lambda: automator.request_stop(),  # Appel a la methode request_stop de l'automator
        style="Custom.TButton",  # Style personnalisé
        state=tk.DISABLED  # Le bouton est désactivé au départ
    )
    stop_button.pack(side=tk.LEFT, padx=5)


    # Menu déroulant pour les modèles
    model_var = tk.StringVar(root)
    model_var.set(DEFAULT_MODEL)  # Valeur par défaut
    model_dropdown = ttk.Combobox(root, textvariable=model_var, values=AVAILABLE_MODELS, state="readonly",
                                 font=(FONT_FAMILY, FONT_SIZE))
    model_dropdown.pack(pady=(0, 5))

    def change_model(event):
        selected_model = model_var.get()
        automator.model = genai.GenerativeModel(selected_model)
        output_text.insert(tk.END, f"Modèle changé pour: {selected_model}\n")
        output_text.see(tk.END)

    model_dropdown.bind("<<ComboboxSelected>>", change_model)

    # Style personnalisé pour les boutons
    style = ttk.Style()
    style.configure(
        "Custom.TButton",
        font=(FONT_FAMILY, FONT_SIZE),
        background=BUTTON_COLOR,
        foreground=BUTTON_TEXT_COLOR,
        borderwidth=0,
        padding=5,
    )
    style.map(
        "Custom.TButton",
        background=[("active", "#81b150"), ("pressed", "#689a3d")],
    )

    # Initialisation de l'automator avec le text widget, le label de status et le bouton envoyer et le bouton stop
    automator = TaskAutomator(api_key, output_text, status_label, send_button, stop_button)

    def change_api_key():
        new_key = simpledialog.askstring("Changer clé API", "Veuillez entrer votre nouvelle clé API Gemini :")
        if new_key:
            if save_api_key(new_key):
                messagebox.showinfo("Changement clé API", "Votre clé API a bien été enregistrée.")
                genai.configure(api_key=new_key)
                automator.model = genai.GenerativeModel("gemini-pro-vision")  # Recharger le model
                output_text.insert(tk.END, f"Nouvelle clé api chargée\n")
                output_text.see(tk.END)
            else:
                messagebox.showerror("Changement clé API",
                                     "Une erreur s'est produite lors de la sauvegarde de la nouvelle clé API.")

    api_key_button = ttk.Button(
        root,
        text="Changer la clé API",
        command=change_api_key,
        style="Custom.TButton"
    )
    api_key_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()