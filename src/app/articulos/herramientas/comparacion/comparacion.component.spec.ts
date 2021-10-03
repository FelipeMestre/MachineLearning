import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ComparacionComponent } from './comparacion.component';

describe('ComparacionComponent', () => {
  let component: ComparacionComponent;
  let fixture: ComponentFixture<ComparacionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ComparacionComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ComparacionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
