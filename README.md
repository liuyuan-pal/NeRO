#   N e R O   f o r   B l e n d e r   D a t a s e t  
 N e R O :   N e u r a l   G e o m e t r y   a n d   B R D F   R e c o n s t r u c t i o n   o f   R e f l e c t i v e   O b j e c t s   f r o m   M u l t i v i e w   I m a g e s  
 ! [ ] ( a s s e t s / t e a s e r . j p g )  
  
 # #   [ P r o j e c t   p a g e ] ( h t t p s : / / l i u y u a n - p a l . g i t h u b . i o / N e R O / )   |   [ P a p e r ] ( h t t p s : / / a r x i v . o r g / a b s / 2 3 0 5 . 1 7 3 9 8 )  
  
 # #   U s a g e  
 # # #   S e t u p  
 1 .   I n s t a l l   b a s i c   r e q u i r e d   p a c k a g e s .  
 ` ` ` s h e l l  
 g i t   c l o n e   h t t p s : / / g i t h u b . c o m / l i u y u a n - p a l / N e R O . g i t  
 c d   N e R O  
 p i p   i n s t a l l   - r   r e q u i r e m e n t s . t x t  
 ` ` `  
 2 .   I n s t a l l   ` n v d i f f r a s t ` .   P l e a s e   f o l l o w   i n s t r u c t i o n s   h e r e   [ h t t p s : / / n v l a b s . g i t h u b . i o / n v d i f f r a s t / # i n s t a l l a t i o n ] ( h t t p s : / / n v l a b s . g i t h u b . i o / n v d i f f r a s t / # i n s t a l l a t i o n ) .  
 3 .   I n s t a l l   ` r a y t r a c i n g ` .   P l e a s e   f o l l o w   i n s t r u c t i o n s   h e r e   [ h t t p s : / / g i t h u b . c o m / a s h a w k e y / r a y t r a c i n g ] ( h t t p s : / / g i t h u b . c o m / a s h a w k e y / r a y t r a c i n g ) .  
  
 # # #   D o w n l o a d   d a t a s e t s  
  
 -   N e R O   M o d e l s   a n d   d a t a s e t s   a l l   c a n   b e   f o u n d   [ h e r e ] ( h t t p s : / / c o n n e c t h k u h k - m y . s h a r e p o i n t . c o m / : f : / g / p e r s o n a l / y u a n l y _ c o n n e c t _ h k u _ h k / E v N z _ o 6 S u E 1 M s X e V y B 0 V o Q 0 B 9 z L 8 N Z X j Q Q g 0 K k n I h 6 R K j Q ? e = M a o n K e ) .  
 -   T h e   * * B l e n d e r * *   d a t a s e t   u s e d   f o r   t e s t i n g   c o m e s   f r o m   [ N e R F ] ( h t t p s : / / d r i v e . g o o g l e . c o m / d r i v e / f o l d e r s / 1 2 8 y B r i W 1 I G _ 3 N J 5 R p 7 A P S T Z s J q d J d f c 1 )   a n d   [ R e f - N e R F ] ( h t t p s : / / s t o r a g e . g o o g l e a p i s . c o m / g r e s e a r c h / r e f r a w 3 6 0 / r e f . z i p ) .  
  
 # # #   C o n f i g   F i l e s  
  
 T h i s   p r o j e c t   i s   a l s o   c o m p a t i b l e   w i t h   t h e   N e R O   d a t a s e t .   I n   t h i s   p r o j e c t ,   w e   h a v e   p l a c e d   t h e   s e t t i n g   o f   ` d a t a s e t _ d i r `   i n   t h e   ` . y a m l `   f i l e .    
  
 >   F o r   e x a m p l e ,   i f   t h e   p a t h   o f   t h e   l e g o   d a t a s e t   i s   ` ~ / N e R O / d a t a / n e r f _ s y n t h e t i c / l e g o ` ,   t h e n   i n   t h e   ` . y a m l `   f i l e ,   ` d a t a s e t _ d i r :   ~ / N e R O / d a t a / n e r f _ s y n t h e t i c ` .  
  
 T o   r u n   B l e n d e r   D a t a s e t ,   w e   n e e d   t o   c h a n g e   t h e   ` . y a m l `   f i l e s .   T h e   ` . y a m l `   f i l e   f o r   t h e   B l e n d e r   d a t a s e t   d i f f e r s   f r o m   N e R O   i n   * * t w o   w a y s * * .    
  
 -   T h e   c a t e g o r y   o f   d a t a b a s e _ n a m e   i s   n e r f .    
 -   I t   i s   n e c e s s a r y   t o   a d d   ` i s _ n e r f :   t r u e `   i n   t h e   ` . y a m l `   f i l e .  
  
 E x a m p l e s   o f   t h e   ` . y a m l `   f i l e s   c a n   b e   s e e n   i n   ` c o n f i g s / s h a p e / n e r f `   a n d   ` c o n f i g s / m a t e r i a l / n e r f ` .  
  
 # # #   S t a g e   I :   S h a p e   r e c o n s t r u c t i o n  
  
 1 .   I n   t h e   ` N e R O `   d i r e c t o r y ,   e n s u r e   t h a t   y o u   h a v e   t h e   f o l l o w i n g   d a t a :  
 ` ` `  
 N e R O  
 | - -   d a t a  
         | - -   G l o s s y R e a l  
                 | - -   b e a r    
                         . . .  
         | - -   G l o s s y S y n t h e t i c  
                 | - -   b e l l  
                  
         | - -   B l e n d e r  
         	 | - -   l e g o  
                         . . .  
 ` ` `  
 2 .   R u n   t h e   t r a i n i n g   s c r i p t  
 ` ` ` s h e l l  
 #   r e c o n s t r u c t i n g   t h e   " b e l l "   o f   t h e   G l o s s y   S y n t h e t i c   d a t a s e t  
 p y t h o n   r u n _ t r a i n i n g . p y   - - c f g   c o n f i g s / s h a p e / s y n / b e l l . y a m l  
  
 #   r e c o n s t r u c t i n g   t h e   " b e a r "   o f   t h e   G l o s s y   R e a l   d a t a s e t  
 p y t h o n   r u n _ t r a i n i n g . p y   - - c f g   c o n f i g s / s h a p e / r e a l / b e a r . y a m l  
  
 #   r e c o n s t r u c t i n g   t h e   " l e g o "   o f   t h e   B l e n d e r   d a t a s e t  
 p y t h o n   r u n _ t r a i n i n g . p y   - - c f g   c o n f i g s / s h a p e / n e r f / l e g o . y a m l  
 ` ` `  
 I n t e r m e d i a t e   r e s u l t s   w i l l   b e   s a v e d   a t   ` d a t a / t r a i n _ v i s ` .   M o d e l s   w i l l   b e   s a v e d   a t   ` d a t a / m o d e l ` .  
  
 3 .   E x t r a c t   m e s h   f r o m   t h e   m o d e l .  
 ` ` ` s h e l l  
 p y t h o n   e x t r a c t _ m e s h . p y   - - c f g   c o n f i g s / s h a p e / s y n / b e l l . y a m l  
 p y t h o n   e x t r a c t _ m e s h . p y   - - c f g   c o n f i g s / s h a p e / r e a l / b e a r . y a m l  
 p y t h o n   e x t r a c t _ m e s h . p y   - - c f g   c o n f i g s / s h a p e / n e r f / l e g o . y a m l  
 ` ` `  
 T h e   e x t r a c t e d   m e s h e s   w i l l   b e   s a v e d   a t   ` d a t a / m e s h e s ` .  
  
 # # #   S t a g e   I I :   M a t e r i a l   e s t i m a t i o n  
  
 1 .   I n   t h e   ` N e R O `   d i r e c t o r y ,   e n s u r e   t h a t   y o u   h a v e   t h e   f o l l o w i n g   d a t a :  
 ` ` `  
 N e R O  
 | - -   d a t a  
         | - -   G l o s s y R e a l  
                 | - -   b e a r    
                         . . .  
         | - -   G l o s s y S y n t h e t i c  
                 | - -   b e l l  
                  
         | - -   B l e n d e r  
         	 | - -   l e g o  
                         . . .  
         | - -   m e s h e s  
                 |   - -   b e l l _ s h a p e - 3 0 0 0 0 0 . p l y  
                 |   - -   b e a r _ s h a p e - 3 0 0 0 0 0 . p l y  
                 |   - -   l e g o _ s h a p e - 3 0 0 0 0 0 . p l y  
                           . . .  
 ` ` `  
 2 .   R u n   t h e   t r a i n i n g   s c r i p t :  
 ` ` ` s h e l l  
 #   e s t i m a t e   B R D F   o f   t h e   " b e l l "   o f   t h e   G l o s s y   S y n t h e t i c   d a t a s e t  
 p y t h o n   r u n _ t r a i n i n g . p y   - - c f g   c o n f i g s / m a t e r i a l / s y n / b e l l . y a m l  
  
 #   e s t i m a t e   B R D F   o f   t h e   " b e a r "   o f   t h e   G l o s s y   R e a l   d a t a s e t  
 p y t h o n   r u n _ t r a i n i n g . p y   - - c f g   c o n f i g s / m a t e r i a l / r e a l / b e a r . y a m l  
  
 #   e s t i m a t e   B R D F   o f   t h e   " l e g o "   o f   t h e   B l e n d e r   d a t a s e t  
 p y t h o n   r u n _ t r a i n i n g . p y   - - c f g   c o n f i g s / m a t e r i a l / n e r f / l e g o . y a m l  
 ` ` `  
 I n t e r m e d i a t e   r e s u l t s   w i l l   b e   s a v e d   a t   ` d a t a / t r a i n _ v i s ` .   M o d e l s   w i l l   b e   s a v e d   a t   ` d a t a / m o d e l ` .  
  
 3 .   E x t r a c t   m a t e r i a l s   f r o m   t h e   m o d e l .  
 ` ` ` s h e l l  
 p y t h o n   e x t r a c t _ m a t e r i a l s . p y   - - c f g   c o n f i g s / m a t e r i a l / s y n / b e l l . y a m l  
 p y t h o n   e x t r a c t _ m a t e r i a l s . p y   - - c f g   c o n f i g s / m a t e r i a l / r e a l / b e a r . y a m l  
 p y t h o n   e x t r a c t _ m a t e r i a l s . p y   - - c f g   c o n f i g s / m a t e r i a l / n e r f / l e g o . y a m l  
 ` ` `  
 T h e   e x t r a c t e d   m a t e r i a l s   w i l l   b e   s a v e d   a t   ` d a t a / m a t e r i a l s ` .  
  
 # # #   R e l i g h t i n g   ( N o t   T e s t i n g )  
  
 1 .   I n   t h e   ` N e R O `   d i r e c t o r y ,   e n s u r e   t h a t   y o u   h a v e   t h e   f o l l o w i n g   d a t a :  
 ` ` `  
 N e R O  
 | - -   d a t a  
         | - -   G l o s s y R e a l  
                 | - -   b e a r    
                         . . .  
         | - -   G l o s s y S y n t h e t i c  
                 | - -   b e l l  
                         . . .  
         | - -   m e s h e s  
                 |   - -   b e l l _ s h a p e - 3 0 0 0 0 0 . p l y  
                 |   - -   b e a r _ s h a p e - 3 0 0 0 0 0 . p l y  
                           . . .  
         | - -   m a t e r i a l s  
                 |   - -   b e l l _ m a t e r i a l - 1 0 0 0 0 0  
                         |   - -   a l b e d o . n p y  
                         |   - -   m e t a l l i c . n p y  
                         |   - -   r o u g h n e s s . n p y  
                 |   - -   b e a r _ m a t e r i a l - 1 0 0 0 0 0  
                         |   - -   a l b e d o . n p y  
                         |   - -   m e t a l l i c . n p y  
                         |   - -   r o u g h n e s s . n p y  
         | - -   h d r  
                 |   - -   n e o n _ p h o t o s t u d i o _ 4 k . e x r  
 ` ` `  
 2 .   R u n   r e l i g h t i n g   s c r i p t  
 ` ` ` s h e l l  
 p y t h o n   r e l i g h t . p y   - - b l e n d e r   < p a t h - t o - y o u r - b l e n d e r >   \  
                                     - - n a m e   b e l l - n e o n   \  
                                     - - m e s h   d a t a / m e s h e s / b e l l _ s h a p e - 3 0 0 0 0 0 . p l y   \  
                                     - - m a t e r i a l   d a t a / m a t e r i a l s / b e l l _ m a t e r i a l - 1 0 0 0 0 0   \  
                                     - - h d r   d a t a / h d r / n e o n _ p h o t o s t u d i o _ 4 k . e x r   \  
                                     - - t r a n s  
                                      
 p y t h o n   r e l i g h t . p y   - - b l e n d e r   < p a t h - t o - y o u r - b l e n d e r >   \  
                                     - - n a m e   b e a r - n e o n   \  
                                     - - m e s h   d a t a / m e s h e s / b e a r _ s h a p e - 3 0 0 0 0 0 . p l y   \  
                                     - - m a t e r i a l   d a t a / m a t e r i a l s / b e a r _ m a t e r i a l - 1 0 0 0 0 0   \  
                                     - - h d r   d a t a / h d r / n e o n _ p h o t o s t u d i o _ 4 k . e x r  
 ` ` `  
 T h e   r e l i g h t i n g   r e s u l t s   w i l l   b e   s a v e d   a t   ` d a t a / r e l i g h t `   w i t h   t h e   d i r e c t o r y   n a m e   o f   ` b e l l - n e o n `   o r   ` b e a r - n e o n ` .   T h i s   c o m m a n d   m e a n s   t h a t   w e   u s e   ` n e o n _ p h o t o s t u d i o _ 4 k . e x r `   t o   r e l i g h t   t h e   o b j e c t .  
  
  
 # # #   T r a i n i n g   o n   c u s t o m   o b j e c t s  
  
 R e f e r   t o   [ c u s t o m _ o b j e c t . m d ] ( c u s t o m _ o b j e c t . m d ) .  
  
 # # #   E v a l u a t i o n  
  
 R e f e r   t o   [ e v a l . m d ] ( e v a l . m d ) .  
  
  
  
 # #   R e s u l t s   ( o n   B l e n d e r   D a t a s e t s )  
  
 # # #   S t a g e 1 :   S h a p e   r e c o n s t r u c t i o n  
  
 |   N e R F             |   P S N R           |   R e f - N e R F   |   P S N R           |  
 |   - - - - - - - - -   |   - - - - - - - -   |   - - - - - - - -   |   - - - - - - - -   |  
 |   c h a i r           |   2 7 . 7 3 9 1 8   |   b a l l           |   3 9 . 6 2 9 6 6   |  
 |   d r u m s           |   2 1 . 0 6 2 8 6   |   c a r             |   2 6 . 0 9 8 7 8   |  
 |   f i c u s           |   2 2 . 5 1 3 1 7   |   c o f f e e       |   3 0 . 6 1 4 0 1   |  
 |   h o t d o g         |   2 9 . 3 3 4 5 1   |   h e l m e t       |   2 9 . 5 6 5 7 3   |  
 |   l e g o             |   2 3 . 4 7 7 4 6   |   t e a p o t       |   3 5 . 4 1 2 3 4   |  
 |   m a t e r i a l s   |   2 4 . 3 2 3 2 3   |   t o a s t e r     |   2 5 . 2 3 6 4 7   |  
 |   m i c               |   2 4 . 5 4 5 1 2   |                     |                     |  
 |   s h i p             |   2 2 . 9 1 3 3 6   |                     |                     |  
  
 # # #   S t a g e 2 :   M a t e r i a l   e s t i m a t i o n  
  
 |   N e R F             |   P S N R           |   R e f - N e R F   |   P S N R           |  
 |   - - - - - - - - -   |   - - - - - - - -   |   - - - - - - - -   |   - - - - - - - -   |  
 |   c h a i r           |   2 8 . 7 4 8 4 7   |   b a l l           |   3 3 . 6 6 3 3 8   |  
 |   d r u m s           |   2 4 . 8 8 2 2 7   |   c a r             |   2 6 . 9 7 6 2     |  
 |   f i c u s           |   2 8 . 3 8 0 8 5   |   c o f f e e       |   3 3 . 7 6 2 3 7   |  
 |   h o t d o g         |   3 2 . 1 3 4 7 5   |   h e l m e t       |   2 9 . 5 9 0 4 4   |  
 |   l e g o             |   2 5 . 6 6 0 8 1   |   t e a p o t       |   4 0 . 2 8 7 3 1   |  
 |   m a t e r i a l s   |   2 4 . 8 5 1 4     |   t o a s t e r     |   2 7 . 3 0 6 6 4   |  
 |   m i c               |   2 8 . 6 3 9 6 3   |                     |                     |  
 |   s h i p             |   2 6 . 5 4 5 9 7   |                     |                     |  
  
 # #   A c k n o w l e d g e m e n t s  
 I n   t h i s   r e p o s i t o r y ,   w e   h a v e   u s e d   c o d e s   f r o m   t h e   f o l l o w i n g   r e p o s i t o r i e s .    
 W e   t h a n k   a l l   t h e   a u t h o r s   f o r   s h a r i n g   g r e a t   c o d e s .  
  
 -   [ N e u S ] ( h t t p s : / / g i t h u b . c o m / T o t o r o 9 7 / N e u S )  
 -   [ N v D i f f R a s t ] ( h t t p s : / / g i t h u b . c o m / N V l a b s / n v d i f f r a s t )  
 -   [ N v D i f f R e c ] ( h t t p s : / / g i t h u b . c o m / N V l a b s / n v d i f f r e c )  
 -   [ R e f - N e R F ] ( h t t p s : / / g i t h u b . c o m / g o o g l e - r e s e a r c h / m u l t i n e r f )  
 -   [ R a y T r a c i n g ] ( h t t p s : / / g i t h u b . c o m / a s h a w k e y / r a y t r a c i n g )  
 -   [ C O L M A P ] ( h t t p s : / / c o l m a p . g i t h u b . i o / )  
  
 # #   C i t a t i o n  
 ` ` `  
 @ i n p r o c e e d i n g s { l i u 2 0 2 3 n e r o ,  
     t i t l e = { N e R O :   N e u r a l   G e o m e t r y   a n d   B R D F   R e c o n s t r u c t i o n   o f   R e f l e c t i v e   O b j e c t s   f r o m   M u l t i v i e w   I m a g e s } ,  
     a u t h o r = { L i u ,   Y u a n   a n d   W a n g ,   P e n g   a n d   L i n ,   C h e n g   a n d   L o n g ,   X i a o x i a o   a n d   W a n g ,   J i e p e n g   a n d   L i u ,   L i n g j i e   a n d   K o m u r a ,   T a k u   a n d   W a n g ,   W e n p i n g } ,  
     b o o k t i t l e = { S I G G R A P H } ,  
     y e a r = { 2 0 2 3 }  
 }  
 ` ` `  
 