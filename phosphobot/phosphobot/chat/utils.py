from phosphobot import __version__


def ascii_test_tube() -> str:
    return f"""
                                                  
                                 [grey46](((((%%(([/grey46]        
                               [grey46](((((((([/grey46][white]&&&&[/white][grey46](([/grey46]     
                            [grey46](((((([/grey46][white]&&&[/white][grey46](((([/grey46][white]&&&&&[/white][grey46](*[/grey46]  
                          [grey46](((((([/grey46][white]&&&&&&&[/white][grey46]((((([/grey46][white]&&&&[/white][grey46]([/grey46] 
                       [grey46](((((([/grey46][white]&&&&&&&&&&&&[/white][grey46]((((((((([/grey46]
                     [grey46](((((([/grey46][white]&&&&&&&&&&&&&&&&&[/white][grey46](((((([/grey46]
                  [grey46](((((([/grey46][bright_green]###(////////////[/bright_green][white]%&[/white][grey46](((((([/grey46]  
                [grey46](((((([/grey46][bright_green]#################[/bright_green][grey46](((((([/grey46]     
             [grey46](((((([/grey46][bright_green]##################[/bright_green][grey46](((((([/grey46]       
           [grey46](((((([/grey46][bright_green]#################[/bright_green][grey46](((((([/grey46]          
        [grey46](((((([/grey46][bright_green]#####(///###///###[/bright_green][grey46](((((([/grey46]      [green]phosphobot chat[/green]
      [grey46](((((([/grey46][bright_green]#################[/bright_green][grey46](((((([/grey46]         [green]{__version__}[/green]
   [grey46](((((([/grey46][bright_green]#####/############[/bright_green][grey46](((((([/grey46]           [green]Copyright (c) 2025 phospho[/green]
  [grey46]((((([/grey46][bright_green]##########////###[/bright_green][grey46](((((([/grey46]              [green]https://phospho.ai[/green]
 [grey46](((([/grey46][bright_green]#####(/##########[/bright_green][grey46](((((([/grey46]                      
 [grey46](((([/grey46][bright_green]####(///######[/bright_green][grey46](((((([/grey46]                         
 [grey46]((((([/grey46][bright_green]###########[/bright_green][grey46](((((([/grey46]                           
  [grey46](((((([/grey46][bright_green]######[/bright_green][grey46](((((([/grey46]                              
    [grey46](((((((((((((([/grey46]                                
    """


KEYBOARD_CONTROl_TEXT = """
[bold green]🎮 Keyboard Control Commands:[/bold green]

Movement:
  ↑ ↓ ← →  Move Forward/Back/Left/Right
  D C      Move Up/Down

Gripper:
  Space    Toggle Open/Close

Mode:
  Ctrl+T   Toggle AI/Keyboard control  
  Ctrl+S   Stop Agent

[dim]Press keys to control the robot immediately[/dim]
"""
